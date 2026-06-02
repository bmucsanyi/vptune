import dataclasses
import importlib.resources
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, override

import autobatch
import pytest
import torch
from torch.nn.utils import parametrize
from vptune_test_helpers import (
    assert_tree_close,
    thresholds_for_measurements,
)

import vptune as vp
import vptune.adapters as vpa
import vptune.ext as vpx
from vptune import autobatch_bridge
from vptune.checks import validate_thresholds
from vptune.data import FullSizeRecord, Measurement
from vptune.errors import ReferenceFailedError
from vptune.identities import (
    canonical_json,
    module_identity,
    stable_hash,
    to_json_value,
)
from vptune.io import read_record, write_record
from vptune.measure import (
    CPUMemoryBackend,
    measure_once,
    measure_operation,
    run_candidate,
)
from vptune.schemas import record_current
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
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=policy,
        selection_policy=vp.SelectionPolicy(),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def _input_signature(case: str = "test", family: str = "family") -> dict[str, object]:
    operator = vp.gradient(family, f"{case}-objective", aggregation="sum").signature()

    return {
        "case": case,
        "operator": operator,
        "operator_spec_hash": stable_hash(operator),
        "target": {"target": "test", "environment": {}},
        "adapter": {"adapter_id": "tests", "adapter_version": "1"},
    }


def test_environment_signature_captures_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    signature = vpx.environment_signature()

    assert signature["torch"]["version"] == torch.__version__
    assert isinstance(signature["torch"]["config"], str)
    assert signature["determinism"] == {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
    }
    assert signature["backend_flags"]["matmul_precision"] == (
        torch.get_float32_matmul_precision()
    )
    assert signature["backend_flags"]["cuda_matmul_allow_tf32"] is (
        torch.backends.cuda.matmul.allow_tf32
    )
    assert signature["env"]["PYTORCH_CUDA_ALLOC_CONF"] == ("expandable_segments:True")
    assert set(signature) == {
        "python",
        "platform",
        "torch",
        "determinism",
        "backend_flags",
        "cuda",
        "rocm",
        "mps",
        "env",
    }


def test_target_signature_includes_declared_device_identity() -> None:
    signature = cpu_target().signature()

    assert signature["device_signatures"] == (
        {
            "device": "cpu",
            "type": "cpu",
            "index": None,
        },
    )


def test_root_api_all_matches_public_surface() -> None:
    assert tuple(vp.__all__) == (
        "AdmissionError",
        "Batch",
        "BufferTree",
        "Candidate",
        "CheckRecord",
        "CohortAssignment",
        "CohortConstraint",
        "DataProvider",
        "Family",
        "FullSizeRecord",
        "FunctionObjective",
        "MaterializationError",
        "Materializer",
        "Measurement",
        "MeasurementError",
        "NoPassedCandidateError",
        "ObjectiveContext",
        "OperatorSpec",
        "ParameterSurface",
        "ParameterTree",
        "Plan",
        "PlanValidationContext",
        "PlanValidator",
        "Problem",
        "ReferenceFailedError",
        "ReferenceResult",
        "ReplayContext",
        "ScalarObjective",
        "SelectionPolicy",
        "StaleRecordError",
        "Target",
        "TensorTree",
        "TimingPolicy",
        "TuningRun",
        "VPTuneError",
        "VectorProvider",
        "autotune",
        "composition",
        "empirical_fisher_vp",
        "fisher_vp",
        "ggnvp",
        "gradient",
        "hvp",
        "inverse_metric",
        "jvp",
        "load_plan",
        "load_tuned_plan",
        "load_tuned_run",
        "materialize",
        "metric",
        "parameter_surface",
        "standard_problem",
        "tune",
        "tune_run",
        "validate_plan",
        "vjp",
    )


def test_extension_api_all_matches_extension_surface() -> None:
    assert tuple(vpx.__all__) == (
        "STANDARD_THRESHOLDS",
        "AnchorRegistry",
        "AutobatchDomain",
        "AutobatchFind",
        "AxisDescriptor",
        "AxisRegistry",
        "CPUMemoryBackend",
        "CUDAMemoryBackend",
        "CallableMaterializer",
        "Candidate",
        "CandidateAdmitter",
        "CandidateOperation",
        "CheckRecord",
        "CohortAssignment",
        "CohortConstraint",
        "CompositionChild",
        "FullSizeRecord",
        "KFACMetricBlock",
        "KFACMetricOperator",
        "MaterializerCallback",
        "Measurement",
        "MemoryBackend",
        "OperationFactory",
        "ReferenceCheck",
        "RuntimeConfig",
        "StandardMetricOperator",
        "admit_checkpoint",
        "admit_forward_ad",
        "admit_functional_call",
        "admit_torch_func",
        "candidate_record_from_json",
        "candidate_record_to_json",
        "check_record_current",
        "check_record_from_json",
        "check_record_to_json",
        "checkpoint_operation",
        "clear_parameter_gradients",
        "composition_operation_factory",
        "composition_reference_check",
        "composition_runtime_config",
        "constant_operation",
        "default_memory_backend",
        "dense_jacobian_anchor",
        "dense_metric_inner",
        "dense_metric_inverse_multiply",
        "dense_metric_inverse_residual",
        "dense_metric_multiply",
        "device_signature",
        "empirical_fisher_vp_dense_anchor",
        "environment_signature",
        "finite_difference_hvp",
        "finite_difference_jvp",
        "fisher_vp_dense_anchor",
        "forward_ad_jvp_anchor",
        "full_size_record_current",
        "full_size_record_from_json",
        "full_size_record_to_json",
        "ggnvp_dense_anchor",
        "gradient_anchor",
        "hvp_anchor",
        "hvp_jvp_grad_anchor",
        "hvp_reverse_over_reverse_anchor",
        "jvp_anchor",
        "module_functional_call",
        "plan_from_json",
        "plan_record_current",
        "plan_to_json",
        "select_fastest_candidate_with_autobatch",
        "selected_plan_validation_summary_current",
        "selected_plan_validation_summary_record",
        "settings_product",
        "standard_axis_descriptors",
        "standard_axis_registry",
        "standard_operation_factory",
        "standard_reference_check",
        "standard_runtime_config",
        "tensor_signature",
        "tree_add",
        "tree_l2_norm",
        "tree_reference_check",
        "tree_zeros_like",
        "vhp_anchor",
        "vjp_anchor",
        "vjp_dot_identity_error",
    )
    assert isinstance(vpx.CPUMemoryBackend(), vpx.CPUMemoryBackend)
    assert vpx.STANDARD_THRESHOLDS["max_abs_diff"] == pytest.approx(1e-4)


def test_extension_tensor_and_measurement_helpers() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.sum().backward()
    tree = {"value": torch.tensor([3.0, 4.0])}
    zeros = vpx.tree_zeros_like(tree)
    summed = vpx.tree_add(tree, zeros)

    vpx.clear_parameter_gradients((parameter,))

    assert parameter.grad is None
    assert vpx.tensor_signature(parameter)["requires_grad"] is True
    assert torch.equal(tree_leaves(zeros)[0], torch.zeros(2))
    assert torch.equal(tree_leaves(summed)[0], tree["value"])
    assert float(vpx.tree_l2_norm(tree)) == pytest.approx(5.0)


def test_cohort_constraint_rejects_unsupported_modes() -> None:
    with pytest.raises(RuntimeError, match="dependency inheritance"):
        vp.CohortConstraint(
            name="bad",
            settings_keys=("model_dtype",),
            assignments=({"model_dtype": "float32"},),
            dependency_inheritance="all_families",
        )

    with pytest.raises(RuntimeError, match="selection aggregation"):
        vp.CohortConstraint(
            name="bad",
            settings_keys=("model_dtype",),
            assignments=({"model_dtype": "float32"},),
            selection_aggregation="mean_elapsed_seconds",
        )


def materialize_candidate_impl(
    candidate: vp.Candidate,
    record: vp.FullSizeRecord,
) -> vpx.CandidateOperation:
    def operation() -> torch.Tensor:
        return torch.tensor([float(candidate.settings.get("scale", 1.0))])

    assert record.candidate_id == candidate.candidate_id

    return operation


materialize_candidate = vpx.CallableMaterializer(
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
    family_input_signatures = {
        family: record.input_signature for family, record in plan.records.items()
    }

    for record in plan.check_records:
        family_input_signatures.setdefault(record.family, record.input_signature)

    return vp.ReplayContext(
        input_signature=plan.input_signature,
        family_input_signatures=family_input_signatures,
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
    saved_full_size_rows = tuple(
        vpx.full_size_record_from_json(read_record(path))
        for path in sorted((run_dir / "full_size").rglob("*.json"))
    )
    saved_check_rows = tuple(
        vpx.check_record_from_json(read_record(path))
        for path in sorted((run_dir / "references").rglob("*.json"))
    )
    full_size_by_key = {
        canonical_json(record.row_key()): record for record in saved_full_size_rows
    }
    check_by_key = {
        canonical_json(record.row_key()): record for record in saved_check_rows
    }
    full_size_rows = tuple(
        full_size_by_key[canonical_json(record.row_key())]
        for record in plan.full_size_records
    )
    check_rows = tuple(
        check_by_key[canonical_json(record.row_key())] for record in plan.check_records
    )

    return full_size_rows, check_rows


def saved_candidate_rows(
    run_dir: Path,
    _: vp.Plan,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        read_record(path) for path in sorted((run_dir / "candidates").rglob("*.json"))
    )


def candidate_records_for_plan(plan: vp.Plan) -> tuple[Mapping[str, object], ...]:
    def candidate_for_record(record: vp.FullSizeRecord) -> vp.Candidate:
        selected_candidate = plan.selected.get(record.family)

        if (
            selected_candidate is not None
            and selected_candidate.candidate_id == record.candidate_id
            and to_json_value(selected_candidate.settings)
            == to_json_value(record.candidate_settings)
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
        vpx.candidate_record_to_json(
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
    row = vpx.candidate_record_to_json(
        candidate,
        _input_signature("migration-source"),
    )
    replayed = vpx.candidate_record_from_json(row)

    assert row["migration_source_id"] == "pilot-row-1"
    assert replayed.migration_source_id == "pilot-row-1"
    assert replayed.signature() == candidate.signature()


def test_write_record_rejects_type_specific_missing_fields(tmp_path: Path) -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    row = vpx.candidate_record_to_json(candidate, _input_signature("schema"))
    missing_status = dict(row)
    missing_status.pop("status")

    with pytest.raises(vp.VPTuneError, match="status"):
        write_record(tmp_path / "missing-status.json", missing_status)

    summary = {
        "record_type": "summary",
        "schema_version": row["schema_version"],
        "package_version": row["package_version"],
        "input_signature": {},
        "candidate_settings": {},
        "status": "passed",
        "generator_id": "plan",
        "generator_version": row["package_version"],
        "selected": {},
        "records": {},
        "full_size_records": (),
        "check_records": (),
        "validation_records": (),
        "validation_required": False,
        "validation_order": (),
        "validator_identities": {},
        "dependencies_by_family": {},
        "cohort_assignment": None,
        "cohort_constraints": (),
        "selected_dependency_identities": {},
        "materializer_identities": {},
        "runtime_identities": {},
        "adapter_identities": {},
        "policy": dataclasses.asdict(vp.SelectionPolicy()),
    }

    with pytest.raises(vp.VPTuneError, match="target_identity"):
        write_record(tmp_path / "missing-target.json", summary)


def test_stable_hash_changes_on_identity_inputs() -> None:
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
    assert vpa.RankStatus
    assert vpa.PilotReadiness
    assert vpa.load_transformers_model
    assert vpa.transformers_attention_axis
    assert vpa.transformers_attn_implementation


def test_package_exposes_type_marker() -> None:
    marker = importlib.resources.files("vptune").joinpath("py.typed")

    assert marker.is_file()


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


def test_module_identity_records_nested_parametrizations() -> None:
    class ExpParametrization(torch.nn.Module):
        @override
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.exp()

    class NegParametrization(torch.nn.Module):
        @override
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return -value

    class NestedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(1, 1, bias=False)

    first = NestedModule()
    second = NestedModule()
    parametrize.register_parametrization(first.block, "weight", ExpParametrization())
    parametrize.register_parametrization(second.block, "weight", NegParametrization())

    first_identity = module_identity(first)
    second_identity = module_identity(second)

    assert first_identity["parametrizations"][0]["parameter"] == "block.weight"
    parametrization_name = first_identity["parametrizations"][0]["parametrizations"][0]
    assert parametrization_name.endswith("ExpParametrization")
    assert first_identity["parametrizations"] != second_identity["parametrizations"]


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

    gradient = vpx.gradient_anchor(scalar, params)
    expected_gradient = 3.0 * params.pow(2) + params

    assert torch.allclose(gradient, expected_gradient)
    assert torch.allclose(
        vpx.jvp_anchor(function, params, vector),
        vpx.finite_difference_jvp(function, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert torch.allclose(
        vpx.forward_ad_jvp_anchor(function, params, vector),
        vpx.jvp_anchor(function, params, vector),
        atol=1e-12,
    )
    assert vpx.vjp_dot_identity_error(function, params, vector, cotangent) < 1e-12
    assert torch.allclose(
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        vpx.finite_difference_hvp(scalar, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert torch.allclose(
        vpx.hvp_anchor(scalar, params, vector),
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        atol=1e-12,
    )
    assert torch.allclose(
        vpx.vhp_anchor(scalar, params, vector),
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
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

    gradient = vpx.gradient_anchor(scalar, params)
    hvp = vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector)

    assert isinstance(gradient, dict)
    assert isinstance(hvp, dict)
    assert torch.allclose(gradient["active"], torch.tensor([4.0], dtype=torch.float64))
    assert torch.allclose(
        gradient["disconnected"],
        torch.zeros(1, dtype=torch.float64),
    )
    assert torch.allclose(hvp["active"], torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(hvp["disconnected"], torch.zeros(1, dtype=torch.float64))


def test_forward_ad_jvp_anchor_supports_module_functional_call() -> None:
    module = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    params = {"weight": torch.tensor([[1.0, -2.0]], dtype=torch.float64)}
    vector = {"weight": torch.tensor([[0.5, 1.5]], dtype=torch.float64)}
    inputs = torch.tensor([[3.0, 4.0]], dtype=torch.float64)

    def function(active_params: dict[str, torch.Tensor]) -> torch.Tensor:
        output = vpx.module_functional_call(
            module,
            active_params,
            {},
            inputs,
            module_mode="eval",
            tie_weights=True,
            strict=False,
            parametrization_policy="active",
            mutates_state=False,
            mutated_parameter_keys=(),
            mutated_buffer_keys=(),
        )
        assert isinstance(output, torch.Tensor)

        return output

    result = vpx.forward_ad_jvp_anchor(function, params, vector)
    expected = inputs @ vector["weight"].T

    assert torch.allclose(result, expected)


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

    assert torch.allclose(vpx.dense_jacobian_anchor(function, params), jacobian)
    assert torch.allclose(
        vpx.ggnvp_dense_anchor(function, loss_hessian, params, vector),
        expected_ggn,
    )

    score_gradients = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    expected_fisher = score_gradients.T @ (score_gradients @ vector) / 3.0

    assert torch.allclose(
        vpx.fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )
    assert torch.allclose(
        vpx.empirical_fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )

    metric = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    inverse_product = vpx.dense_metric_inverse_multiply(metric, vector)

    assert torch.allclose(vpx.dense_metric_multiply(metric, vector), metric @ vector)
    assert torch.allclose(vpx.dense_metric_multiply(metric, inverse_product), vector)
    assert torch.allclose(
        vpx.dense_metric_inner(metric, vector, vector),
        vector @ (metric @ vector),
    )
    assert vpx.dense_metric_inverse_residual(metric, inverse_product, vector) < 1e-12


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

        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"backend_id": "tests.recording_memory"}

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

    record = FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status=status,
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=samples,
        memory_samples=samples,
        dependency_identities=dict(candidate.dependency_identities),
        reference_passed=reference_passed,
    )

    return record


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
        dependency_identities=dict(candidate.dependency_identities),
    )

    return record


def _identity_kwargs(
    families: tuple[str, ...] = ("family",),
) -> dict[str, Any]:
    return {
        "target_identity": {"target": "test", "environment": {}},
        "runtime_identities": {
            family: {"runtime": f"test.{family}"} for family in families
        },
        "adapter_identities": {
            family: {"adapter_id": "tests", "adapter_version": "1"}
            for family in families
        },
    }


def test_plan_materialize_validates_selected_dependency_identities() -> None:
    input_signature = {"case": "plan-materialize-dependencies"}
    dependency = vp.Candidate("dependency", "selected", {}, admission_status="passed")
    dependent = vp.Candidate(
        "dependent",
        "selected",
        {},
        dependency_identities={"dependency": {"stale": "identity"}},
        admission_status="passed",
    )
    dependency_record = _record(
        dependency,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    dependent_record = _record(
        dependent,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vp.Plan(
        selected={
            "dependency": dependency,
            "dependent": dependent,
        },
        records={
            "dependency": dependency_record,
            "dependent": dependent_record,
        },
        input_signature=input_signature,
        policy=vp.SelectionPolicy(),
        materializers={
            "dependency": materialize_candidate,
            "dependent": materialize_candidate,
        },
        dependencies_by_family={
            "dependency": (),
            "dependent": ("dependency",),
        },
    )

    with pytest.raises(vp.MaterializationError, match="dependency identity"):
        plan.materialize("dependent")


def test_validate_plan_materializes_in_validation_order() -> None:
    input_signature = {"case": "validation-materialization-order"}
    calls = []

    def first_materializer_callback(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
    ) -> str:
        assert candidate.family == record.family
        calls.append("materialize:first")

        return "first"

    def second_materializer_callback(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
    ) -> str:
        assert candidate.family == record.family
        calls.append("materialize:second")

        return "second"

    first_materializer = vpx.CallableMaterializer(
        "tests.first_materializer",
        "1",
        {},
        first_materializer_callback,
    )
    second_materializer = vpx.CallableMaterializer(
        "tests.second_materializer",
        "1",
        {},
        second_materializer_callback,
    )
    first = vp.Candidate("first", "row", {}, admission_status="passed")
    first_record = _record(
        first,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    first_identity = {
        "family": "first",
        "candidate_id": "row",
        "candidate_settings": dict(first.settings),
        "full_size_row": first_record.row_key(),
        "materializer_identity": dict(first_materializer.identity()),
    }
    second = vp.Candidate(
        "second",
        "row",
        {},
        dependency_identities={"first": first_identity},
        admission_status="passed",
    )
    second_record = _record(
        second,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vp.Plan(
        selected={"first": first, "second": second},
        records={"first": first_record, "second": second_record},
        input_signature=input_signature,
        policy=vp.SelectionPolicy(),
        materializers={
            "first": first_materializer,
            "second": second_materializer,
        },
        validation_order=("first", "second"),
        dependencies_by_family={"first": (), "second": ("first",)},
    )

    def first_validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.family == record.family
        assert context.selected == "first"
        calls.append("validate:first")
        message = "stop before downstream materialization"
        raise RuntimeError(message)

    def second_validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.family == record.family
        assert context.selected == "second"
        calls.append("validate:second")

        return reference_passed()

    with pytest.raises(RuntimeError, match="downstream materialization"):
        vp.validate_plan(
            plan,
            {"first": first_validator, "second": second_validator},
        )

    assert calls == ["materialize:first", "validate:first"]


def _current_record(record: FullSizeRecord) -> FullSizeRecord:
    return record


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


def test_selection_rejects_record_from_different_candidate_settings() -> None:
    signature = {"case": "current"}
    candidate = vp.Candidate("family", "row", {})
    mismatched = dataclasses.replace(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
        ),
        candidate_settings={"axis": "different"},
    )

    with pytest.raises(vp.NoPassedCandidateError):
        select_family(
            ((candidate, mismatched),),
            input_signature=signature,
            policy=vp.SelectionPolicy(),
        )


def test_selection_uses_json_normalized_signatures() -> None:
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


def test_reference_row_key_includes_row_and_check_identity() -> None:
    candidate_a = vp.Candidate("family", "row-a", {"axis": "same"})
    candidate_b = vp.Candidate("family", "row-b", {"axis": "same"})
    first = _check_record(candidate_a, input_signature={})
    second = _check_record(candidate_b, input_signature={})
    third = dataclasses.replace(first, name="second")

    assert first.row_key() != second.row_key()
    assert first.row_key() != third.row_key()


def test_admission_helpers() -> None:
    vpx.admit_functional_call({
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
        vpx.admit_torch_func({
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
        vpx.admit_checkpoint({
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
    output = vpx.checkpoint_operation(
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


def test_checkpoint_operation_can_disable_rng_preservation() -> None:
    values = []
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "checkpoint_policy": "non_reentrant_no_rng_preservation",
            **checkpoint_fields(),
            "preserve_rng_state": False,
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(value)
        values.append(noise.detach().clone())

        return (value * noise).sum()

    torch.manual_seed(17)
    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="checkpoint_policy",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert len(values) == 2
    assert not torch.equal(values[0], values[1])


def test_checkpoint_operation_rejects_no_rng_policy_with_rng_preservation() -> None:
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "checkpoint_policy": "non_reentrant_no_rng_preservation",
            **checkpoint_fields(),
        },
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square()

    with pytest.raises(vp.AdmissionError, match="preserve_rng_state=False"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="checkpoint_policy",
        )()


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
        vpx.checkpoint_operation(
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
    ) -> vpx.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        def function(value: torch.Tensor) -> torch.Tensor:
            return value * 3.0

        return vpx.checkpoint_operation(
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

    runtime = vpx.RuntimeConfig(
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

    standard_factory = vpx.standard_operation_factory(
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
            "pytorch_sdpa_direct",
            "patched_eager",
            "transformers_eager",
            "transformers_sdpa",
            "transformers_flash_attention_2",
            "transformers_flash_attention_3",
            "transformers_flash_attention_4",
            "paged|flash_attention_2",
            "paged|flash_attention_3",
            "paged|flash_attention_4",
        ),
        policy=policy,
    )
    flash = vp.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_flash = vp.Candidate(
        "family",
        "float-flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "model_dtype": "float32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash3 = vp.Candidate(
        "family",
        "flash3",
        {
            "attention.frontend": "transformers_flash_attention_3",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash4 = vp.Candidate(
        "family",
        "flash4",
        {
            "attention.frontend": "transformers_flash_attention_4",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    paged_flash = vp.Candidate(
        "family",
        "paged-flash",
        {
            "attention.frontend": "paged|flash_attention_4",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    attentions = vp.Candidate(
        "family",
        "attentions",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": True,
        },
    )
    math_attention = vp.Candidate(
        "family",
        "math",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    math_attentions = vp.Candidate(
        "family",
        "math-attentions",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": True,
        },
    )
    direct_flash_kernel = vp.Candidate(
        "family",
        "sdpa-flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_direct_flash_kernel = vp.Candidate(
        "family",
        "float-sdpa-flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
            "model_dtype": "float32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    priority_sdpa = vp.Candidate(
        "family",
        "priority-sdpa",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    bad_priority_sdpa = vp.Candidate(
        "family",
        "bad-priority-sdpa",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("priority_list",),
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    eval_dropout = vp.Candidate(
        "family",
        "dropout",
        {
            "attention.frontend": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.1,
        },
    )
    bad_gqa = vp.Candidate(
        "family",
        "gqa",
        {
            "attention.frontend": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "enable_gqa": True,
            "query_heads": 5,
            "key_value_heads": 2,
        },
    )
    valid_gqa = vp.Candidate(
        "family",
        "valid-gqa",
        {
            "attention.frontend": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.0,
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
            "attention.frontend": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.0,
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
            "attention.frontend": "patched_eager",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": True,
            "patched_attention_id": "gemma-softcap-eager",
            "patched_attention_semantics": {"logit_softcap": 30.0},
        },
    )
    untracked_patch = vp.Candidate(
        "family",
        "untracked-patch",
        {
            "attention.frontend": "patched_eager",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )

    assert axis.admit(flash) == (True, None)
    assert axis.admit(float_flash)[0] is False
    assert axis.admit(flash3) == (True, None)
    assert axis.admit(flash4) == (True, None)
    assert axis.admit(paged_flash) == (True, None)
    assert axis.admit(attentions)[0] is False
    assert axis.admit(math_attention) == (True, None)
    assert axis.admit(math_attentions)[0] is False
    assert axis.admit(direct_flash_kernel) == (True, None)
    assert axis.admit(float_direct_flash_kernel)[0] is False
    assert axis.admit(priority_sdpa) == (True, None)
    assert axis.admit(bad_priority_sdpa)[0] is False
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
                {"attention.frontend": "unknown"},
            ),
            policy=policy,
        )[0]
        is False
    )

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attention_axis(("unknown",), policy=policy)


def test_axis_registry_admits_grid_and_records_failed_admission() -> None:
    model = torch.nn.Linear(1, 1)
    registry = vpx.AxisRegistry()

    def admit_dtype(candidate: vp.Candidate) -> tuple[bool, str | None]:
        if candidate.settings["dtype"] == "float16":
            return False, "float16 disabled"

        return True, None

    registry.register(
        vpx.AxisDescriptor(
            "dtype",
            ("dtype",),
            ("float32", "float16"),
            admission_rule=admit_dtype,
        )
    )

    with pytest.raises(vp.AdmissionError):
        registry.register(vpx.AxisDescriptor("other", ("dtype",), ("float64",)))

    candidates = vpx.settings_product(
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
    ) -> vpx.CandidateOperation:
        assert candidate.settings["dtype"] == "float32"
        assert batch["family"] == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

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
        runtime=vpx.RuntimeConfig(
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
    registry = vpx.AxisRegistry()
    registry.register(
        vpx.AxisDescriptor(
            "pair",
            ("left", "right"),
            ({"left": 1, "right": 2},),
        )
    )
    candidates = vpx.settings_product(
        "family",
        {"pair": ({"left": 1, "right": 2},)},
        axis_registry=registry,
    )

    assert candidates[0].settings == {"left": 1, "right": 2}
    assert candidates[0].changed_axes == ("pair",)
    assert registry.admit(candidates[0]).admission_status == "passed"

    with pytest.raises(vp.AdmissionError):
        vpx.settings_product(
            "family",
            {"pair": ({"left": 1},)},
            axis_registry=registry,
        )


def test_standard_axis_registry_validates_core_axes() -> None:
    registry = vpx.standard_axis_registry()
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
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "math",
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
    valid_forward_ad = vp.Candidate(
        "family",
        "valid-forward-ad",
        {
            "operator_path": "forward_ad_jvp",
            "requires_forward_ad": True,
            "forward_ad_supported": True,
        },
    )
    unsupported_forward_ad = vp.Candidate(
        "family",
        "unsupported-forward-ad",
        {
            "operator_path": "forward_ad_jvp",
            "requires_forward_ad": True,
            "forward_ad_supported": False,
        },
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
    invalid_no_rng_policy = vp.Candidate(
        "family",
        "invalid-no-rng-policy",
        {
            "checkpoint_policy": "non_reentrant_no_rng_preservation",
            **checkpoint_fields,
        },
    )
    valid_no_rng_policy = vp.Candidate(
        "family",
        "valid-no-rng-policy",
        {
            "checkpoint_policy": "non_reentrant_no_rng_preservation",
            **checkpoint_fields,
            "preserve_rng_state": False,
        },
    )

    assert registry.admit(candidate).admission_status == "passed"
    assert registry.admit(bad_budget).admission_status == "failed"
    assert registry.admit(bad_flag).admission_status == "failed"
    assert registry.admit(bad_dtype).admission_status == "failed"
    assert registry.admit(bad_path).admission_status == "failed"
    assert registry.admit(missing_torch_func_fields).admission_status == "failed"
    assert registry.admit(valid_torch_func).admission_status == "passed"
    assert registry.admit(valid_forward_ad).admission_status == "passed"
    assert registry.admit(unsupported_forward_ad).admission_status == "failed"
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
    assert registry.admit(invalid_no_rng_policy).admission_status == "failed"
    assert registry.admit(valid_no_rng_policy).admission_status == "passed"
    assert registry.axes["model_dtype"].admit(bad_dtype)[0] is False

    with pytest.raises(vp.AdmissionError):
        vpx.AxisRegistry().register(vpx.AxisDescriptor("bad", ("x",), ()))


def test_problem_signature_includes_axis_registry_identity() -> None:
    model = torch.nn.Linear(1, 1)
    first_registry = vpx.AxisRegistry()
    second_registry = vpx.AxisRegistry()
    first_registry.register(
        vpx.AxisDescriptor(
            "axis",
            ("axis",),
            ("value",),
            adapter_id="adapter",
            adapter_version="1",
        )
    )
    second_registry.register(
        vpx.AxisDescriptor(
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
    ) -> vpx.CandidateOperation:
        assert candidate
        assert batch

        return vpx.constant_operation(vector)

    first = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vpx.RuntimeConfig(
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
            {
                "model_dtype": "float32",
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
            },
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked",
            {"model_dtype": "bfloat16"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked-frontend",
            {
                "model_dtype": "float32",
                "attention.frontend": "transformers_flash_attention_2",
            },
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked-kernel",
            {
                "model_dtype": "float32",
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "flash_attention",
            },
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
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "allowed"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    target = vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float32",),
        allowed_attention_frontends=("pytorch_sdpa_direct",),
        allowed_sdpa_kernels=("math",),
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
        runtime=vpx.RuntimeConfig(
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
    failed_rows = tuple(row for row in plan.full_size_records if row.status == "failed")
    assert len(failed_rows) == 3
    assert all(row.error_type == "AdmissionError" for row in failed_rows)


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

    selected = vpx.select_fastest_candidate_with_autobatch(
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
    ) -> vpx.CandidateOperation:
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
        runtime=vpx.RuntimeConfig(
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
                vpx.AutobatchDomain(
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


def test_plan_replay_preserves_autobatch_selected_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
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
        assert goal == autobatch.Goal.largest_safe()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
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
        runtime=vpx.RuntimeConfig(
            (vp.Candidate("family", "base", {}, admission_status="passed"),),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "autobatch-domain"},
            (
                vpx.AutobatchDomain(
                    axis_name="batch_size",
                    values=(1, 2),
                    settings_by_value={
                        1: {"batch_size": 1},
                        2: {"batch_size": 2},
                    },
                    value_to_settings_id="tests.batch_size_settings",
                    admission_identity={"case": "test"},
                    goal="largest_safe",
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
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 11.0)),
    )

    assert plan.selected["family"].candidate_id == "base|batch_size=2"

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={"family": materialize_candidate},
    )

    assert replayed.selected["family"].candidate_id == "base|batch_size=2"


def test_plan_replay_rejects_unsupported_selection_policy(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.replay-policy"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    context = dataclasses.replace(
        replay_context_for_plan(plan),
        selection_policy=vp.SelectionPolicy(speed_statistic="mean_elapsed_seconds"),
    )

    with pytest.raises(vp.VPTuneError, match="unsupported speed statistic"):
        vpx.plan_from_json(
            read_record(tmp_path / "summaries" / "tuning.json"),
            replay_context=context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=saved_candidate_rows(tmp_path, plan),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_rejects_changed_memory_backend_identity(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.memory-backend"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    changed_input = dict(plan.input_signature)
    changed_input["measurement"] = {
        "memory_backend": {"backend_id": "tests.changed_memory_backend"}
    }
    family_signatures = dict(replay_context_for_plan(plan).family_input_signatures)
    family_signatures["family"] = changed_input
    context = dataclasses.replace(
        replay_context_for_plan(plan),
        input_signature=changed_input,
        family_input_signatures=family_signatures,
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)

    assert plan.input_signature["measurement"]["memory_backend"] == dict(
        CPUMemoryBackend().identity()
    )

    with pytest.raises(vp.StaleRecordError, match="input signature"):
        vpx.plan_from_json(
            read_record(tmp_path / "summaries" / "tuning.json"),
            replay_context=context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=saved_candidate_rows(tmp_path, plan),
            materializers={"family": materialize_candidate},
        )


def test_tune_rejects_adapter_identity_that_contradicts_runtime() -> None:
    model = torch.nn.Linear(1, 1)

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vpx.RuntimeConfig(
            (vp.Candidate("family", "row", {}, admission_status="passed"),),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {
                "runtime": "test.adapter-runtime",
                "adapter_id": "adapter.test",
                "adapter_version": "1",
            },
        ),
    )

    with pytest.raises(vp.MaterializationError, match="adapter identity"):
        vp.tune(problem, memory_backend=CPUMemoryBackend())


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
    ) -> vpx.CandidateOperation:
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
        runtime=vpx.RuntimeConfig(
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
    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={"family": materialize_candidate},
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

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


def test_tune_builds_measured_operation_before_clock(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    events = []
    clock_values = iter((0.0, 1.0))

    def clock() -> float:
        events.append("clock")

        return next(clock_values)

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)
        events.append("build")

        def operation() -> torch.Tensor:
            events.append("run")

            return vector

        return operation

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

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
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "unit_test.build-before-clock"},
        ),
    )

    vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=clock,
    )

    assert events == ["build", "clock", "run", "clock"]


def test_tune_writes_admission_failure_rows_without_measurement(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    failed_candidate = vp.Candidate(
        "family",
        "blocked",
        {},
        admission_status="failed",
        admission_error="blocked by admission",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate
        assert batch
        assert vector
        message = "reference check should not run"
        raise AssertionError(message)

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate
        assert batch
        assert vector
        message = "operation should not run"
        raise AssertionError(message)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vpx.RuntimeConfig(
            (failed_candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "admission-failure"},
        ),
    )

    with pytest.raises(vp.NoPassedCandidateError):
        vp.tune(
            problem,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    candidate_row = read_record(
        tmp_path / "candidates" / "family" / "blocked" / "candidate.json"
    )
    check_row = read_record(
        tmp_path / "references" / "family" / "blocked" / "tree_close.json"
    )
    full_size_row = read_record(
        tmp_path / "full_size" / "family" / "blocked" / "result.json"
    )

    assert candidate_row["status"] == "failed"
    assert check_row["status"] == "failed"
    assert check_row["error_type"] == "AdmissionError"
    assert full_size_row["status"] == "failed"
    assert full_size_row["reference_passed"] is False


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
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "good"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

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
        runtime=vpx.RuntimeConfig(
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
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"

        return vpx.constant_operation(vector)

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
        runtime=vpx.RuntimeConfig(
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
    full_size_row = read_record(
        tmp_path / "full_size" / "family" / "row" / "result.json"
    )
    reference_row = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close.json"
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")

    assert read_record(tmp_path / "candidates" / "family" / "row" / "candidate.json")
    candidate_row = read_record(
        tmp_path / "candidates" / "family" / "row" / "candidate.json"
    )
    replayed_candidate = vpx.candidate_record_from_json(candidate_row)

    assert replayed_candidate == candidate

    changed_candidate_row = dict(candidate_row)
    changed_candidate_row["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(full_size_row),),
            check_records=(vpx.check_record_from_json(reference_row),),
            candidate_records=(changed_candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    assert record_current(
        full_size_row,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
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
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert vpx.plan_record_current(summary, plan)

    stale_summary = dict(summary)
    stale_summary["generator_version"] = "stale"

    assert not vpx.plan_record_current(stale_summary, plan)

    changed_full_size = dict(full_size_row)
    changed_full_size["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(changed_full_size),),
            check_records=(vpx.check_record_from_json(reference_row),),
            candidate_records=(candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    changed_reference = dict(reference_row)
    changed_reference["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(full_size_row),),
            check_records=(vpx.check_record_from_json(changed_reference),),
            candidate_records=(candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    replayed = vpx.plan_from_json(
        summary,
        replay_context=replay_context_for_plan(plan),
        full_size_records=(vpx.full_size_record_from_json(full_size_row),),
        check_records=(vpx.check_record_from_json(reference_row),),
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
    ) -> vpx.CandidateOperation:
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
        runtime=vpx.RuntimeConfig(
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
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "anchor"
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    def candidate_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        if candidate.candidate_id == "bad":
            return vpx.constant_operation(vector * 2.0)

        return vpx.constant_operation(vector)

    check = vpx.tree_reference_check(
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
    input_signature = _input_signature("json")
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
        input_signature=input_signature,
    )
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )
    row_path = tmp_path / "full_size.json"
    plan_path = tmp_path / "plan.json"

    write_record(row_path, vpx.full_size_record_to_json(record))
    write_record(plan_path, vpx.plan_to_json(plan))

    loaded = read_record(row_path)
    round_tripped = vpx.full_size_record_from_json(loaded)
    loaded_plan = read_record(plan_path)

    assert round_tripped == record
    assert vpx.plan_record_current(loaded_plan, plan)

    stale_row = dict(loaded)
    stale_row["input_signature"] = {"case": "changed"}

    assert not record_current(
        stale_row,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )

    stale_plan = dict(loaded_plan)
    stale_plan["input_signature"] = {"case": "changed"}

    assert not vpx.plan_record_current(stale_plan, plan)


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
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

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
        runtime=vpx.RuntimeConfig(
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
        vpx.plan_from_json(
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
        vpx.plan_from_json(
            mismatched_summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            run_dir=tmp_path,
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={},
            run_dir=tmp_path,
        )


def test_plan_replay_rejects_stale_context_and_materializer() -> None:
    input_signature = _input_signature("replay-context")
    other_signature = _input_signature("other")
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
            input_signature=input_signature,
        )
    )
    check = _check_record(candidate, input_signature=input_signature)
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )
    summary = vpx.plan_to_json(plan)
    context = replay_context_for_plan(plan)
    other_materializer = vpx.CallableMaterializer(
        "tests.other_materializer",
        "1",
        {},
        materialize_candidate_impl,
    )

    assert vpx.plan_to_json(plan) != vpx.plan_to_json(
        dataclasses.replace(
            plan,
            materializers={"family": other_materializer},
        )
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                input_signature=other_signature,
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                family_input_signatures={"family": other_signature},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
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
        vpx.plan_from_json(
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
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                target_identity={"target": "other", "environment": {}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
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
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                adapter_identities={
                    "family": {"adapter_id": "other", "adapter_version": "1"}
                },
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": other_materializer},
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(),
            materializers={"family": materialize_candidate},
        )

    stale_candidate_row = dict(candidate_records_for_plan(plan)[0])
    stale_candidate_row["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(stale_candidate_row,),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_recomputes_family_selection() -> None:
    input_signature = _input_signature("recompute")
    memory_signature = _input_signature("recompute-memory")
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
            input_signature=input_signature,
        )
    )
    slow_record = _current_record(
        _record(
            slow,
            elapsed=(2.0,),
            reserved=(1.0,),
            input_signature=input_signature,
        )
    )
    fast_check = _check_record(fast, input_signature=input_signature)
    slow_check = _check_record(slow, input_signature=input_signature)
    plan = vp.Plan(
        selected={"family": slow},
        records={"family": slow_record},
        input_signature=input_signature,
        policy=vp.SelectionPolicy(),
        full_size_records=(fast_record, slow_record),
        check_records=(fast_check, slow_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
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
            input_signature=memory_signature,
        )
    )
    high_memory_record = _current_record(
        _record(
            high_memory,
            elapsed=(1.0,),
            reserved=(10.0,),
            input_signature=memory_signature,
        )
    )
    low_memory_check = _check_record(
        low_memory,
        input_signature=memory_signature,
    )
    high_memory_check = _check_record(
        high_memory,
        input_signature=memory_signature,
    )
    memory_plan = vp.Plan(
        selected={"family": high_memory},
        records={"family": high_memory_record},
        input_signature=memory_signature,
        policy=vp.SelectionPolicy(),
        full_size_records=(low_memory_record, high_memory_record),
        check_records=(low_memory_check, high_memory_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(memory_plan),
            replay_context=replay_context_for_plan(memory_plan),
            full_size_records=(low_memory_record, high_memory_record),
            check_records=(low_memory_check, high_memory_check),
            candidate_records=candidate_records_for_plan(memory_plan),
            materializers={"family": materialize_candidate},
        )


def test_selected_plan_validation_writes_failed_record(tmp_path: Path) -> None:
    input_signature = _input_signature("validation")
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
        input_signature=input_signature,
    )
    check = _check_record(candidate, input_signature=input_signature)
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
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
        tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
    )
    summary = read_record(tmp_path / "summaries" / "selected_plan_validation.json")

    assert failed["status"] == "failed"
    assert failed["error_type"] == "ReferenceFailedError"
    assert summary["status"] == "failed"
    failed_record = vpx.check_record_from_json(failed)

    assert summary["records"] == [failed_record.row_key()]

    assert vpx.selected_plan_validation_summary_current(
        summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
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

    assert not vpx.selected_plan_validation_summary_current(
        stale_summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_summary=stale_summary,
            validation_records=(failed_record,),
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
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
    input_signature = _input_signature("validation-runtime")
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
        input_signature=input_signature,
    )
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
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
        tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
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
    assert operator.batch_inputs == {
        "reference": ("symmetry_vector",),
        "operation": (),
    }
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
    assert fisher.batch_inputs == {"reference": (), "operation": ()}
    empirical = vp.empirical_fisher_vp(
        "metric",
        "retain",
        aggregation="mean",
        loss_reduction="per_example",
        denominator="num_examples",
    )

    assert empirical.kind == "empirical_fisher_vp"
    assert empirical.semantics == {
        "loss_reduction": "per_example",
        "denominator": "num_examples",
    }
    assert empirical.batch_inputs == {"reference": (), "operation": ()}
    ggn = vp.ggnvp(
        "metric",
        "retain",
        aggregation="mean",
        loss_geometry="psd_metric",
    )
    linear_ggn = vp.ggnvp(
        "metric",
        "retain",
        aggregation="mean",
        loss_geometry="linear_map",
    )

    assert ggn.kind == "ggnvp"
    assert ggn.batch_inputs == {
        "reference": ("loss_hessian", "symmetry_vector"),
        "operation": ("loss_hessian",),
    }
    assert linear_ggn.batch_inputs == {
        "reference": ("loss_hessian",),
        "operation": ("loss_hessian",),
    }
    assert vp.gradient("grad", "loss", aggregation="sum").kind == "gradient"
    jvp = vp.jvp("jvp", "function", aggregation="none")
    vjp = vp.vjp("vjp", "function", aggregation="none")
    metric = vp.metric("metric", "retain", aggregation="mean")
    inverse_metric = vp.inverse_metric("inverse_metric", "retain", aggregation="mean")

    assert jvp.kind == "jvp"
    assert jvp.batch_inputs == {"reference": (), "operation": ()}
    assert vjp.kind == "vjp"
    assert vjp.batch_inputs == {"reference": ("tangent_vector",), "operation": ()}
    assert metric.kind == "metric"
    assert metric.batch_inputs == {
        "reference": ("metric",),
        "operation": ("metric",),
    }
    assert inverse_metric.kind == "inverse_metric"
    assert inverse_metric.batch_inputs == {
        "reference": ("metric",),
        "operation": ("metric",),
    }
    assert (
        vp.composition("compose", "hvp_after_metric", aggregation="none").kind
        == "composition"
    )


def test_tune_run_preflight_errors_do_not_write_summary(tmp_path: Path) -> None:
    target = cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = vp.gradient("a", "loss_a", aggregation="sum")
    operator_b = vp.gradient("b", "loss_b", aggregation="sum")

    def make_problem(operator: vp.OperatorSpec) -> vp.Problem:
        candidate = vp.Candidate(operator.family, "row", {}, admission_status="passed")

        def reference_check(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.ReferenceResult:
            assert candidate
            assert batch
            assert vector
            message = "reference check should not run"
            raise AssertionError(message)

        def operation_factory(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vpx.CandidateOperation:
            assert candidate
            assert batch
            assert vector
            message = "operation should not run"
            raise AssertionError(message)

        return vp.Problem(
            model=model,
            params=vp.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=vpx.RuntimeConfig(
                (candidate,),
                operation_factory,
                reference_check,
                materialize_candidate,
                None,
                {"generator": operator.family},
            ),
        )

    cases = (
        (
            tmp_path / "missing-problems",
            vp.TuningRun(
                target=target,
                families=(vp.Family("a", operator_a),),
                run_id="missing-problems",
            ),
            "run adapter is required",
        ),
        (
            tmp_path / "duplicate-problems",
            vp.TuningRun(
                target=target,
                families=(vp.Family("a", operator_a),),
                problems=(make_problem(operator_a), make_problem(operator_a)),
                run_id="duplicate-problems",
            ),
            "problem operator families must be unique",
        ),
        (
            tmp_path / "family-mismatch",
            vp.TuningRun(
                target=target,
                families=(vp.Family("b", operator_b),),
                problems=(make_problem(operator_a),),
                run_id="family-mismatch",
            ),
            "run families must match",
        ),
    )

    for run_dir, run, message in cases:
        with pytest.raises(vp.MaterializationError, match=message):
            vp.tune_run(
                run,
                run_dir=run_dir,
                memory_backend=CPUMemoryBackend(),
                clock=SequenceClock(()),
            )

        assert not (run_dir / "summaries" / "tuning.json").exists()


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
        ) -> vpx.CandidateOperation:
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
            runtime=vpx.RuntimeConfig(
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
        "candidate_settings": dict(plan.selected["a"].settings),
        "full_size_row": plan.records["a"].row_key(),
        "materializer_identity": materialize_candidate.identity(),
    }

    assert dependency_identity == expected_dependency_identity
    assert plan.records["b"].dependency_identities["a"] == expected_dependency_identity
    assert plan.selected_dependency_identities() == {
        "a": {},
        "b": {"a": expected_dependency_identity},
    }

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
    )
    loaded_run = vp.load_tuned_run(
        tmp_path,
        run,
        memory_backend=CPUMemoryBackend(),
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)
    assert vpx.plan_record_current(vpx.plan_to_json(loaded_run), plan)

    stale_dependency_identity = dict(expected_dependency_identity)
    stale_dependency_identity["full_size_row"] = {
        **dict(expected_dependency_identity["full_size_row"]),
        "candidate_id": "stale",
    }
    stale_child = dataclasses.replace(
        plan.selected["b"],
        dependency_identities={"a": stale_dependency_identity},
    )
    stale_plan = dataclasses.replace(
        plan,
        selected={"a": plan.selected["a"], "b": stale_child},
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(stale_plan),
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

    validators_by_family = {"a": validator, "b": validator}
    validation_plan = dataclasses.replace(
        plan,
        validation_required=True,
        validator_identities={
            "a": {"validator": "dag-validator"},
            "b": {"validator": "dag-validator"},
        },
    )
    validation_records = tuple(
        vpx.check_record_from_json(vpx.check_record_to_json(record))
        for record in vp.validate_plan(validation_plan, validators_by_family)
    )
    validated_plan = dataclasses.replace(
        validation_plan,
        validation_records=validation_records,
    )
    validation_summary = vpx.selected_plan_validation_summary_record(
        validation_plan,
        validation_records,
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=validation_records,
        )

    replayed_with_validation = vpx.plan_from_json(
        vpx.plan_to_json(validated_plan),
        replay_context=replay_context_for_plan(
            validated_plan,
            validation_required=True,
        ),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(validated_plan),
        materializers=plan.materializers,
        validation_summary=validation_summary,
        validation_records=validation_records,
    )

    assert replayed_with_validation.validation_required
    assert vpx.plan_record_current(
        vpx.plan_to_json(replayed_with_validation),
        validated_plan,
    )

    changed_validation_record = dataclasses.replace(
        validation_records[0],
        candidate_settings={"axis": "changed"},
    )
    changed_validation_records = (
        changed_validation_record,
        *validation_records[1:],
    )

    assert not vpx.selected_plan_validation_summary_current(
        validation_summary,
        validation_plan,
        changed_validation_records,
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(validated_plan),
            replay_context=replay_context_for_plan(
                validated_plan,
                validation_required=True,
            ),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(validated_plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=changed_validation_records,
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(validated_plan),
            replay_context=replay_context_for_plan(
                validated_plan,
                validation_required=True,
            ),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(validated_plan),
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
    ) -> vpx.CandidateOperation:
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
        runtime=vpx.RuntimeConfig(
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
    validation_record = vpx.check_record_from_json(
        read_record(
            tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
        )
    )
    validation_summary = read_record(
        tmp_path / "summaries" / "selected_plan_validation.json"
    )
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
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
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
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
    assert to_json_value(
        tuple(record.row_key() for record in plan.validation_records)
    ) == (to_json_value((validation_record.row_key(),)))
    assert to_json_value(
        tuple(record.row_key() for record in replayed.validation_records)
    ) == to_json_value((validation_record.row_key(),))
    assert plan.validator_identities == {
        "family": {"validator_id": "tests.validator.v1"}
    }
    assert validation_summary["status"] == "passed"

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
        )

    forged_record = dataclasses.replace(
        validation_record,
        name="tree_close",
    )
    forged_summary = vpx.selected_plan_validation_summary_record(
        plan,
        (forged_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=forged_summary,
            validation_records=(forged_record,),
        )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

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


def test_tune_run_writes_selected_summary_before_validator_failure(
    tmp_path: Path,
) -> None:
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
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")

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
    ) -> vpx.CandidateOperation:
        assert candidate.family == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return vector

        return operation

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == record.candidate_id
        assert context.family == "family"

        message = "validation failed"
        raise RuntimeError(message)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "validation-failure-run"},
        ),
    )
    run = vp.TuningRun(
        target=target,
        families=(vp.Family("family", operator),),
        problems=(problem,),
        validators={"family": validator},
        validator_identities={"family": {"validator_id": "tests.validator.fail"}},
        run_id="validation-failure-run",
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        vp.tune_run(
            run,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock((0.0, 1.0)),
        )

    tuning_summary = read_record(tmp_path / "summaries" / "tuning.json")
    validation_summary = read_record(
        tmp_path / "summaries" / "selected_plan_validation.json"
    )

    assert tuning_summary["validation_required"] is True
    assert tuning_summary["validation_records"] == []
    assert validation_summary["status"] == "failed"


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
        ) -> vpx.CandidateOperation:
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
            runtime=vpx.RuntimeConfig(
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
        ) -> vpx.CandidateOperation:
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
            runtime=vpx.RuntimeConfig(
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

    assert calls == ["a-first", "b-first", "c-row", "a-second", "b-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert plan.selected["c"].candidate_id == "c-row"
    assert plan.cohort_assignment is not None
    assert plan.cohort_assignment.values == {"backend": "second", "chunk": 2}
    assert plan.selected["b"].dependency_identities["a"]["candidate_id"] == "a-second"

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    candidate_rows = candidate_records_for_plan(plan)
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_rows,
        materializers=plan.materializers,
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

    changed_cohort_record = dataclasses.replace(
        saved_full_size[0],
        cohort_assignment={"assignment_id": "stale"},
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(changed_cohort_record, *saved_full_size[1:]),
            check_records=saved_checks,
            candidate_records=candidate_rows,
            materializers=plan.materializers,
        )

    candidates_by_key = {
        (
            candidate.family,
            candidate.candidate_id,
            canonical_json(candidate.settings),
        ): candidate
        for candidate in (
            vpx.candidate_record_from_json(candidate_row)
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
            record.family,
            record.candidate_id,
            canonical_json(record.candidate_settings),
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
        vpx.plan_from_json(
            vpx.plan_to_json(stale_plan),
            replay_context=replay_context_for_plan(stale_plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_rows,
            materializers=plan.materializers,
        )


def test_tune_run_cohort_subset_handles_cross_boundary_dependencies(
    tmp_path: Path,
) -> None:
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
        ) -> vpx.CandidateOperation:
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
            runtime=vpx.RuntimeConfig(
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
            vp.Family("c", operator_c, dependencies=("b",)),
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
                (vp.Candidate("b", "b-row", {}, admission_status="passed"),),
            ),
            make_problem(
                "c",
                operator_c,
                (
                    vp.Candidate(
                        "c",
                        "c-first",
                        {"backend": "first"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "c",
                        "c-second",
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
                families=("a", "c"),
            ),
        ),
        run_id="cohort-cross-boundary-deps",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            11.0,
            11.0,
            21.0,
            21.0,
            22.0,
            22.0,
            23.0,
            23.0,
            24.0,
        )),
    )
    replayed = vp.load_tuned_run(
        tmp_path,
        run,
        memory_backend=CPUMemoryBackend(),
    )

    assert calls == ["a-first", "b-row", "c-first", "a-second", "b-row", "c-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-row"
    assert plan.selected["c"].candidate_id == "c-second"
    assert plan.selected["b"].dependency_identities["a"]["candidate_id"] == "a-second"
    assert plan.selected["c"].dependency_identities["b"]["candidate_id"] == "b-row"
    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)


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
        ) -> vpx.CandidateOperation:
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
            runtime=vpx.RuntimeConfig(
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
            make_problem(
                "c",
                operator_c,
                (
                    vp.Candidate(
                        "c",
                        "c-first",
                        {"backend": "first"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "c",
                        "c-second",
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
        clock=SequenceClock((0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0)),
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

    assert calls == ["c-first", "a-second", "b-second", "c-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert plan.selected["c"].candidate_id == "c-second"
    assert tuple(record.candidate_id for record in failed_references) == ("a-first",)
    assert tuple(record.candidate_id for record in blocked) == ("b-first",)


def test_tune_run_propagates_candidate_validation_errors_inside_cohort(
    tmp_path: Path,
) -> None:
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    model = torch.nn.Linear(1, 1)
    operator = vp.gradient("family", "loss", aggregation="sum")
    candidates = (
        vp.Candidate(
            "family",
            "duplicate",
            {"backend": "first"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "duplicate",
            {"backend": "first"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "valid",
            {"backend": "second"},
            admission_status="passed",
        ),
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
    ) -> vpx.CandidateOperation:
        assert candidate.family == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return vector

        return operation

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "validation-error"},
        ),
    )
    run = vp.TuningRun(
        target=target,
        families=(vp.Family("family", operator),),
        problems=(problem,),
        cohort_constraints=(
            vp.CohortConstraint(
                name="backend",
                settings_keys=("backend",),
                assignments=({"backend": "first"}, {"backend": "second"}),
            ),
        ),
        run_id="cohort-validation-error",
    )

    with pytest.raises(vp.MaterializationError, match="candidate ids must be unique"):
        vp.tune_run(
            run,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock((0.0, 1.0)),
        )
