import contextlib
import dataclasses
import importlib.resources
import inspect
import math
import types
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
import vptune.measure as measure_module
import vptune.run as run_module
from vptune import autobatch_bridge
from vptune.checks import (
    numeric_error_bound_measurements,
    validate_numeric_error_bound,
    validate_thresholds,
)
from vptune.data import FullSizeRecord, Measurement
from vptune.errors import ReferenceFailedError
from vptune.identities import (
    canonical_json,
    cuda_driver_version,
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
from vptune.tensor_tree import (
    tree_add_foreach,
    tree_dot_foreach,
    tree_elementwise_div_foreach,
    tree_elementwise_mul_foreach,
    tree_from_leaves,
    tree_l2_norm_foreach,
    tree_leaves,
    tree_map,
    tree_max_abs_foreach,
    tree_mul_foreach,
    tree_signature,
    tree_sub_foreach,
    tree_zeros_like_foreach,
)


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
        allowed_dtypes=("float64", "fp32", "bf16", "fp16"),
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=policy,
        selection_policy=vp.SelectionPolicy(),
        search_policy=vp.SearchPolicy(strategy="exhaustive"),
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


def test_cuda_driver_version_is_none_when_runtime_does_not_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CUDARTWithoutDriverVersion:
        pass

    def fake_cudart() -> CUDARTWithoutDriverVersion:
        return CUDARTWithoutDriverVersion()

    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        fake_cudart,
    )

    assert cuda_driver_version() is None


def test_target_signature_includes_declared_device_identity() -> None:
    signature = cpu_target().signature()

    assert signature["device_signatures"] == (
        {
            "device": "cpu",
            "type": "cpu",
            "index": None,
        },
    )
    assert signature["search_policy"] == {
        "strategy": "exhaustive",
        "retained_top_count": None,
        "compile_call_horizons": (),
        "variance_repeat_count": None,
    }


def test_search_policy_rejects_unknown_strategy() -> None:
    with pytest.raises(RuntimeError, match="unsupported search strategy"):
        vp.SearchPolicy(strategy="random")


def test_search_policy_requires_balanced_top_count() -> None:
    with pytest.raises(RuntimeError, match="retained_top_count"):
        vp.SearchPolicy(strategy="balanced")


def test_search_policy_requires_thorough_fields() -> None:
    with pytest.raises(RuntimeError, match="compile_call_horizons"):
        vp.SearchPolicy(strategy="thorough", retained_top_count=1)

    with pytest.raises(RuntimeError, match="variance_repeat_count"):
        vp.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(1,),
        )


def test_tune_rejects_thorough_horizon_mismatch_before_probe() -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(),
            selection_policy=vp.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vp.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(2,),
            variance_repeat_count=2,
        ),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-policy"},
        ),
    )

    with pytest.raises(vp.MaterializationError, match="selection horizon"):
        vp.tune(problem)

    assert calls == []


def test_tune_smoke_strategy_measures_baseline_and_class_c_rows(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidates = (
        vp.Candidate("family", "base", {}, admission_status="passed"),
        vp.Candidate(
            "family",
            "hvp-path",
            {"hvp.path": "reverse_over_reverse"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "gradient-graph",
            {"gradient.graph_schedule": "build_once"},
            changed_axes=("gradient.graph_schedule",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "dtype",
            {"dtype.model_compute": "fp32"},
            changed_axes=("dtype.model_compute",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "attention",
            {"attention.frontend": "transformers_sdpa"},
            changed_axes=("attention.frontend",),
            admission_status="passed",
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="smoke"),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=TwoProbeData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-smoke"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp-path",
        "dtype",
    )
    assert tuple(
        row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)
    ) == (
        "attention",
        "base",
        "dtype",
        "gradient-graph",
        "hvp-path",
    )
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp-path",
        "hvp-path",
        "dtype",
        "dtype",
    )
    assert all(call[2]["index"] == 0 for call in calls if call[0] == "operation")
    assert plan.selected_candidate().candidate_id == "base"


def test_tune_smoke_strategy_requires_one_admitted_baseline() -> None:
    model = torch.nn.Linear(1, 1)
    target = dataclasses.replace(
        cpu_target(),
        search_policy=vp.SearchPolicy(strategy="smoke"),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "hvp-path"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "hvp-path"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (
                vp.Candidate(
                    "family",
                    "hvp-path",
                    {"hvp.path": "reverse_over_reverse"},
                    changed_axes=("hvp.path",),
                    admission_status="passed",
                ),
            ),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-smoke"},
        ),
    )

    with pytest.raises(vp.MaterializationError, match="baseline"):
        vp.tune(problem)


def test_tune_fast_strategy_compiles_near_fastest_eager_rows(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    compile_settings = {
        "compile.enabled": "true",
        "compile.boundary": "whole_operator",
        "compile.backend": "inductor",
        "compile.mode": "default",
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": "false",
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": "false",
        "compile.cache_state": "warm_cache",
    }
    candidates = (
        vp.Candidate("family", "base", {}, admission_status="passed"),
        vp.Candidate(
            "family",
            "hvp",
            {"hvp.path": "reverse_over_reverse"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "input",
            {"input.residency": "gpu"},
            changed_axes=("input.residency",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "dtype",
            {"dtype.model_compute": "fp32"},
            changed_axes=("dtype.model_compute",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "compiled-hvp",
            {"hvp.path": "reverse_over_reverse", **compile_settings},
            changed_axes=("hvp.path", "compile.enabled"),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "compiled-dtype",
            {"dtype.model_compute": "fp32", **compile_settings},
            changed_axes=("dtype.model_compute", "compile.enabled"),
            admission_status="passed",
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="fast"),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vp.Candidate,
            inputs: tuple[tuple[vp.Batch, vp.TensorTree], ...],
            output: vp.TensorTree,
            samples: tuple[vp.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.settings.get("compile.enabled") != "true":
                return {}

            return {
                "compile_time_seconds": 0.0,
                "steady_elapsed_seconds": 0.5,
                "recompile_count": 0,
            }

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-fast"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 3.0, 3.0, 5.0, 5.0, 6.0, 6.0, 7.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp",
        "dtype",
        "compiled-dtype",
    )
    assert tuple(
        row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)
    ) == (
        "base",
        "compiled-dtype",
        "compiled-hvp",
        "dtype",
        "hvp",
        "input",
    )
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp",
        "hvp",
        "dtype",
        "dtype",
        "compiled-dtype",
        "compiled-dtype",
    )
    assert plan.selected_candidate().candidate_id == "compiled-dtype"


def test_tune_balanced_strategy_crosses_retained_group_winners(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    compile_settings = {
        "compile.enabled": "true",
        "compile.boundary": "whole_operator",
        "compile.backend": "inductor",
        "compile.mode": "default",
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": "false",
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": "false",
        "compile.cache_state": "warm_cache",
    }
    candidates = (
        vp.Candidate("family", "base", {}, admission_status="passed"),
        vp.Candidate(
            "family",
            "hvp-slow",
            {"hvp.path": "reverse_over_reverse"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "hvp-fast",
            {"hvp.path": "jvp_grad"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "dtype",
            {"dtype.model_compute": "fp32"},
            changed_axes=("dtype.model_compute",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "compiled-cross",
            {"hvp.path": "jvp_grad", "dtype.model_compute": "fp32", **compile_settings},
            changed_axes=(
                "hvp.path",
                "dtype.model_compute",
                "compile.enabled",
            ),
            admission_status="passed",
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "balanced_compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vp.Candidate,
            inputs: tuple[tuple[vp.Batch, vp.TensorTree], ...],
            output: vp.TensorTree,
            samples: tuple[vp.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.settings.get("compile.enabled") != "true":
                return {}

            return {
                "compile_time_seconds": 0.0,
                "steady_elapsed_seconds": 0.3,
                "recompile_count": 0,
            }

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            5.0,
            5.0,
            9.0,
            9.0,
            10.0,
            10.0,
            11.0,
            11.0,
            11.8,
            11.8,
            13.0,
        )),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
    )
    assert {row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)} == {
        "base",
        "hvp-slow",
        "hvp-fast",
        "dtype",
        "compiled-cross",
        "balanced:hvp-fast+dtype",
    }
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp-slow",
        "hvp-fast",
        "hvp-slow",
        "hvp-fast",
        "dtype",
        "dtype",
        "balanced:hvp-fast+dtype",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
        "compiled-cross",
    )
    assert plan.selected_candidate().candidate_id == "compiled-cross"


def test_tune_balanced_strategy_halves_group_rows_by_probe_stage(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidates = (
        vp.Candidate("family", "base", {}, admission_status="passed"),
        vp.Candidate(
            "family",
            "hvp-slow",
            {"hvp.path": "reverse_over_reverse"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "hvp-middle",
            {"hvp.path": "jvp_grad"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "hvp-fast",
            {"hvp.path": "autograd_functional_hvp"},
            changed_axes=("hvp.path",),
            admission_status="passed",
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(vector, torch.Tensor)
        calls.append(("operation", candidate.candidate_id, batch["index"]))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls.append(("reference", candidate.candidate_id, None))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=TwoProbeData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced-halving"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            20.0,
            20.0,
            23.0,
            23.0,
            24.0,
            24.0,
            25.0,
            25.0,
            30.0,
            30.0,
            31.0,
        )),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp-fast",
    )
    assert plan.selected_candidate().candidate_id == "hvp-fast"
    assert tuple(
        call for call in calls if call[0] == "operation" and call[1] != "base"
    ) == (
        ("operation", "hvp-slow", 0),
        ("operation", "hvp-middle", 0),
        ("operation", "hvp-fast", 0),
        ("operation", "hvp-fast", 1),
        ("operation", "hvp-middle", 1),
        ("operation", "hvp-fast", 0),
        ("operation", "hvp-fast", 1),
    )


def test_tune_balanced_strategy_with_only_baseline_measures_it_once(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "base", {}, admission_status="passed")
    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)
        calls.append(("operation", candidate.candidate_id))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls.append(("reference", candidate.candidate_id))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced-baseline"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == ("base",)
    assert tuple(call[0] for call in calls) == ("reference", "operation")
    assert plan.selected_candidate().candidate_id == "base"


def test_tune_thorough_strategy_uses_declared_repeat_count(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "base", {}, admission_status="passed")
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(
                vp.TimingPolicy(
                    short_seconds=0.0,
                    medium_seconds=0.0,
                    long_measured_calls=1,
                )
            ),
            selection_policy=vp.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vp.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(3,),
            variance_repeat_count=2,
        ),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-thorough"},
        ),
    )
    clock = SequenceClock((0.0, 1.0, 1.0, 2.0, 2.0, 3.0))
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=clock,
    )

    assert plan.selected_candidate().candidate_id == "base"
    assert len(plan.records["family"].timing_samples) == 2
    assert tuple(call[1] for call in calls) == ("base", "base")
    assert clock.index == 6


def test_tune_thorough_strategy_records_compile_horizon_scores(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    compile_settings = {
        "compile.enabled": "true",
        "compile.boundary": "whole_operator",
        "compile.backend": "inductor",
        "compile.mode": "default",
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": "false",
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": "false",
        "compile.cache_state": "warm_cache",
    }
    base = vp.Candidate("family", "base", {}, admission_status="passed")
    compiled = vp.Candidate(
        "family",
        "compiled",
        compile_settings,
        changed_axes=("compile.enabled",),
        admission_status="passed",
    )
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(
                vp.TimingPolicy(
                    short_seconds=0.0,
                    medium_seconds=0.0,
                    long_measured_calls=1,
                )
            ),
            selection_policy=vp.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vp.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(3, 6),
            variance_repeat_count=2,
        ),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "thorough_compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vp.Candidate,
            inputs: tuple[tuple[vp.Batch, vp.TensorTree], ...],
            output: vp.TensorTree,
            samples: tuple[vp.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.candidate_id == "base":
                return {}

            assert candidate.candidate_id == "compiled"

            return {
                "compile_time_seconds": 6.0,
                "steady_elapsed_seconds": 2.0,
                "recompile_count": 1,
            }

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id in {"base", "compiled"}
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id in {"base", "compiled"}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (base, compiled),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-thorough-compile-horizons"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            20.0,
            20.0,
            30.0,
            30.0,
            31.0,
            31.0,
            32.0,
            32.0,
            33.0,
        )),
    )
    assert plan.selected_candidate().candidate_id == "compiled"
    metadata = plan.records["family"].selection_metadata
    scores = metadata["compile_amortized_seconds_by_horizon"]
    assert isinstance(scores, Mapping)
    assert scores["3"] == pytest.approx(6.0)
    assert scores["6"] == pytest.approx(4.0)

    saved_records, _ = saved_plan_rows(tmp_path, plan)
    saved_compiled = next(
        record for record in saved_records if record.candidate_id == "compiled"
    )
    saved_scores = saved_compiled.selection_metadata[
        "compile_amortized_seconds_by_horizon"
    ]
    assert isinstance(saved_scores, Mapping)
    assert saved_scores["3"] == pytest.approx(6.0)
    assert saved_scores["6"] == pytest.approx(4.0)


def test_tune_admission_strategy_returns_candidate_table_only(tmp_path: Path) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        "family",
        "row",
        {"dtype.model_compute": "fp32"},
        admission_status="passed",
    )
    target = dataclasses.replace(
        cpu_target(),
        search_policy=vp.SearchPolicy(strategy="admission"),
    )

    def operation_factory(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vp.Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        calls.append(("reference", candidate, batch, vector))

        return reference_passed()

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.admission-search"},
        ),
    )
    plan = vp.tune(problem, run_dir=tmp_path)

    assert plan.selected == {}
    assert plan.records == {}
    assert plan.candidate_rows == (candidate,)
    assert plan.full_size_records == ()
    assert plan.check_records == ()
    assert calls == []
    assert saved_candidate_rows(tmp_path, plan)[0]["candidate_id"] == "row"
    assert not (tmp_path / "references").exists()
    assert not (tmp_path / "full_size").exists()

    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=(),
        check_records=(),
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={},
    )

    assert replayed.selected == {}
    assert tuple(candidate.signature() for candidate in replayed.candidate_rows) == (
        candidate.signature(),
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
        "ModuleCallSpec",
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
        "SearchPolicy",
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
        "sampled_fisher_vp",
        "standard_problem",
        "tune",
        "tune_run",
        "validate_plan",
        "vjp",
    )
    assert issubclass(vp.MeasurementError, vp.VPTuneError)


def test_standard_front_door_signatures_match_spec() -> None:
    assert tuple(inspect.signature(vp.autotune).parameters) == (
        "model",
        "parameter_surface",
        "parameter_values",
        "buffers",
        "data",
        "operator",
        "vectors",
        "target",
        "candidates",
        "thresholds",
        "objective_signature",
        "scalar_objectives",
        "function_objectives",
        "run_dir",
        "memory_backend",
        "clock",
    )
    assert tuple(inspect.signature(vp.standard_problem).parameters) == (
        "model",
        "parameter_surface",
        "parameter_values",
        "buffers",
        "data",
        "operator",
        "vectors",
        "target",
        "candidates",
        "thresholds",
        "objective_signature",
        "scalar_objectives",
        "function_objectives",
    )


def test_extension_api_all_matches_extension_surface() -> None:
    assert tuple(vpx.__all__) == (
        "STANDARD_THRESHOLDS",
        "AnchorRegistry",
        "AttentionInputs",
        "AttentionLocation",
        "AttentionSemantics",
        "AttentionSettings",
        "AutobatchDomain",
        "AutobatchFind",
        "AxisDescriptor",
        "AxisManifest",
        "AxisRegistry",
        "AxisTable",
        "AxisTableAdmitter",
        "AxisTableDescriptor",
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
        "FullSizeCheck",
        "FullSizeRecord",
        "KFACMetricBlock",
        "KFACMetricOperator",
        "MappingAttentionLocation",
        "MaterializerCallback",
        "Measurement",
        "MemoryBackend",
        "ModuleCallSpec",
        "OperationFactory",
        "ReferenceCheck",
        "RuntimeConfig",
        "StandardMetricOperator",
        "admit_checkpoint",
        "admit_core_attention",
        "admit_forward_ad",
        "admit_functional_call",
        "admit_torch_func",
        "apply_final_logit_softcap",
        "apply_softcap",
        "attention_operation_factory",
        "attention_reference_check",
        "attention_settings_from_candidate",
        "axis_manifest",
        "axis_table",
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
        "core_attention_axis",
        "default_memory_backend",
        "dense_jacobian_anchor",
        "dense_metric_inner",
        "dense_metric_inverse_multiply",
        "dense_metric_inverse_residual",
        "dense_metric_multiply",
        "device_signature",
        "empirical_fisher_vp_dense_anchor",
        "environment_signature",
        "exact_attention",
        "execute_attention",
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
        "run_attention",
        "sampled_fisher_vp_dense_anchor",
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
    assert vpx.AxisManifest is vpx.AxisTable
    assert vpx.axis_manifest().signature() == vpx.axis_table().signature()


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


def test_tensor_tree_foreach_inventory_matches_python_ops() -> None:
    left = {
        "a": torch.tensor([1.0, -2.0], dtype=torch.float64),
        "b": torch.tensor([3.0], dtype=torch.float64),
    }
    right = {
        "a": torch.tensor([4.0, 5.0], dtype=torch.float64),
        "b": torch.tensor([-6.0], dtype=torch.float64),
    }
    thresholds = {"max_abs_diff": 1e-12, "max_rel_diff": 1e-12}

    assert_tree_close(
        tree_zeros_like_foreach(left),
        {
            "a": torch.zeros(2, dtype=torch.float64),
            "b": torch.zeros(1, dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_add_foreach(left, right),
        {
            "a": torch.tensor([5.0, 3.0], dtype=torch.float64),
            "b": torch.tensor([-3.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_sub_foreach(left, right),
        {
            "a": torch.tensor([-3.0, -7.0], dtype=torch.float64),
            "b": torch.tensor([9.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_mul_foreach(left, 2.0),
        {
            "a": torch.tensor([2.0, -4.0], dtype=torch.float64),
            "b": torch.tensor([6.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_elementwise_mul_foreach(left, right),
        {
            "a": torch.tensor([4.0, -10.0], dtype=torch.float64),
            "b": torch.tensor([-18.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_elementwise_div_foreach(right, left),
        {
            "a": torch.tensor([4.0, -2.5], dtype=torch.float64),
            "b": torch.tensor([-2.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )

    assert float(tree_dot_foreach(left, right)) == pytest.approx(-24.0)
    assert float(tree_max_abs_foreach(right)) == pytest.approx(6.0)
    assert float(tree_l2_norm_foreach(left)) == pytest.approx(math.sqrt(14.0))

    with pytest.raises(RuntimeError, match="same dtype"):
        tree_add_foreach(left, {"a": right["a"].float(), "b": right["b"]})


def test_axis_table_matches_spec_feature_space() -> None:
    axis_table = vpx.axis_table()
    by_key = axis_table.by_key()
    expected_keys = {
        "activation.offload",
        "activation.recompute",
        "attention.custom_kernel_id",
        "attention.frontend",
        "attention.mask_formatter_id",
        "attention.padding",
        "attention.partition",
        "attention.sdpa_kernel",
        "autocast",
        "batch.data_microbatch_size",
        "batch.empirical_example_batch_size",
        "batch.fisher_sample_batch_size",
        "batch.ggn_batch_size",
        "batch.hvp_row_batch_size",
        "call.buffer_mutation",
        "call.buffers",
        "call.grad_mode",
        "call.params",
        "call.parametrizations",
        "call.path",
        "call.return_type",
        "call.tied_weights",
        "checkpoint.context_fn",
        "checkpoint.determinism_check",
        "checkpoint.early_stop",
        "checkpoint.preserve_rng_state",
        "checkpoint.use_reentrant",
        "chunk.class_block_size_with_exact_global_normalization",
        "chunk.layer_block_size",
        "chunk.lm_head_weight_chunk_bytes",
        "chunk.output_cotangent_block_size",
        "chunk.parameter_block_size",
        "chunk.sequence_position_block_size",
        "chunk.token_block_size",
        "comm.collective_bucket_size",
        "comm.overlap",
        "comm.prefetch",
        "compile.backend",
        "compile.boundary",
        "compile.cache_state",
        "compile.compiled_autograd",
        "compile.cuda_graphs",
        "compile.dynamic",
        "compile.enabled",
        "compile.fullgraph",
        "compile.mode",
        "compile.options.epilogue_fusion",
        "compile.options.shape_padding",
        "composition.child_evaluation",
        "composition.execution",
        "composition.validation",
        "context_parallel.enabled",
        "context_parallel.rotate_method",
        "context_parallel.sequence_dim",
        "distributed.launch",
        "distributed.local_rank_binding",
        "distributed.mesh_dim_names",
        "distributed.mesh_shape",
        "distributed.process_group_backend",
        "distributed.strategy",
        "dtensor.cotangent_placement",
        "dtensor.logits_placement",
        "dtensor.output_placement",
        "dtensor.params_placement",
        "dtensor.redistribute_schedule",
        "dtensor.tangent_placement",
        "dtensor.vector_placement",
        "dtype.accumulation",
        "dtype.autodiff_compute",
        "dtype.intermediate",
        "dtype.metric_factor",
        "dtype.model_compute",
        "dtype.output",
        "dtype.parameter_storage",
        "dtype.vector",
        "empirical_fisher.accumulation",
        "empirical_fisher.grad_path",
        "fisher.accumulation",
        "fisher.expectation_path",
        "fisher.score_grad_path",
        "fsdp.dp_mesh_dims",
        "fsdp.ignored_params",
        "fsdp.mp_policy.cast_forward_inputs",
        "fsdp.mp_policy.output_dtype",
        "fsdp.mp_policy.param_dtype",
        "fsdp.mp_policy.reduce_dtype",
        "fsdp.offload_policy",
        "fsdp.reshard_after_forward",
        "fsdp.shard_placement_fn",
        "fsdp.wrap_granularity",
        "fusion.logits",
        "fusion.loss",
        "fusion.mlp",
        "fusion.norm",
        "fusion.rope",
        "ggn.cotangent_reuse",
        "ggn.jvp_path",
        "ggn.jvp_reuse",
        "ggn.loss_hessian_kernel",
        "ggn.loss_hessian_path",
        "ggn.vjp_path",
        "gradient.graph_schedule",
        "gradient.path",
        "gradient.value_reuse",
        "hvp.gradient_reuse",
        "hvp.graph_schedule",
        "hvp.path",
        "hvp.primal_reuse",
        "input.batch_layout",
        "input.host_to_device",
        "input.length_grouping",
        "input.residency",
        "inverse_metric.factor_reuse",
        "inverse_metric.block_schedule",
        "inverse_metric.iteration_budget",
        "inverse_metric.preconditioner",
        "inverse_metric.solve_path",
        "jvp.linearize_reuse",
        "jvp.path",
        "layout.aliasing",
        "layout.contiguity",
        "layout.flatten_order",
        "layout.output",
        "layout.params",
        "layout.parametrizations",
        "layout.vector",
        "layout.vector_ops",
        "memory.factor_residency",
        "memory.intermediate_residency",
        "memory.jvp_outputs",
        "memory.output_buffers",
        "memory.output_cotangents",
        "memory.primal_outputs",
        "memory.vector_residency",
        "metric.accumulation",
        "metric.block_schedule",
        "metric.multiply_path",
        "numeric.bf16_reduced_precision_reduction",
        "numeric.deterministic_algorithms",
        "numeric.float32_matmul_precision",
        "numeric.fp16_reduced_precision_reduction",
        "numeric.loss_scaling",
        "sampled_fisher.accumulation",
        "sampled_fisher.exact_fisher_check",
        "sampled_fisher.sample_source",
        "sampled_fisher.score_grad_path",
        "schedule.gradient_accumulation",
        "schedule.per_example",
        "schedule.per_token",
        "sequence_parallel.enabled",
        "sequence_parallel.norm_modules",
        "sequence_parallel.output_placement_policy",
        "teacher_outputs",
        "tp.embedding",
        "tp.lm_head",
        "tp.loss_parallel",
        "tp.mlp_down",
        "tp.mlp_up_gate",
        "tp.output_projection",
        "tp.plan",
        "tp.prepare_module_input",
        "tp.prepare_module_output",
        "tp.qkv_projection",
        "vectorization.batch_size",
        "vectorization.in_dims",
        "vectorization.mode",
        "vectorization.randomness",
        "vectorization.vmap_chunk_size",
        "vjp.closure_reuse",
        "vjp.path",
    }
    expected_groups = {
        "activation_memory",
        "ad_lowering",
        "attention_dispatch",
        "compile",
        "distributed_layout",
        "fusion",
        "input_schedule",
        "inverse_solve",
        "metric_storage",
        "numeric_backend",
    }
    grouped_keys = tuple(
        key for keys in axis_table.class_c_groups.values() for key in keys
    )

    assert set(by_key) == expected_keys
    assert set(axis_table.class_c_groups) == expected_groups
    assert set(grouped_keys) == expected_keys
    assert len(grouped_keys) == len(expected_keys)
    assert axis_table.merge_rules == (
        (
            "attention.partition=packed_tokens merges "
            "attention_dispatch with input_schedule"
        ),
        "compile.boundary=attention_module merges compile with attention_dispatch",
        "compile.boundary operator part merges compile with ad_lowering",
        "fusion non-default merges fusion with ad_lowering",
        "dtensor placement merges distributed_layout with ad_lowering",
        "fsdp reduce dtype not fp32 merges distributed_layout with numeric_backend",
        "factorized inverse rows merge inverse_solve with metric_storage",
    )

    for axis in by_key.values():
        assert axis.owner_id
        assert axis.value_domain
        assert axis.admission_rule_id
        assert axis.lowering_rule_id or axis.adapter_id
        assert axis.axis_key in axis_table.class_c_groups[axis.class_c_group]

    assert by_key["compile.mode"].value_domain == (None, "default", "max-autotune")
    assert by_key["attention.frontend"].value_domain == (
        "transformers_eager",
        "transformers_sdpa",
        "transformers_flash_attention_2",
        "transformers_flash_attention_3",
        "transformers_flash_attention_4",
        "transformers_flex_attention",
        "paged|eager",
        "paged|sdpa",
        "paged|flash_attention_2",
        "paged|flash_attention_3",
        "paged|flash_attention_4",
        "registered_transformers_attention",
        "pytorch_sdpa_direct",
        "patched_eager",
        "packed_exact",
        "blockwise_exact",
    )
    assert by_key["metric.multiply_path"].value_domain == (
        "dense_matmul",
        "factorized_multiply",
        "blockwise_multiply",
        "streaming_multiply",
    )
    assert by_key["inverse_metric.solve_path"].value_domain == (
        "dense_solve",
        "cholesky_solve",
        "eigh_solve",
        "svd_solve",
        "conjugate_gradient",
        "factorized_solve",
        "blockwise_solve",
        "woodbury_low_rank_solve",
    )
    assert by_key["layout.vector_ops"].class_a == ("mostly_independent_after_admission")
    assert by_key["teacher_outputs"].class_b == "conditionally_independent"
    assert by_key["dtensor.params_placement"].adapter_id == (
        "vptune.adapters.distributed"
    )
    assert axis_table.signature()["axis_table_version"] == "1"


@pytest.mark.parametrize(
    ("settings", "changed_axes", "groups"),
    [
        (
            {"attention.partition": "packed_tokens"},
            ("attention.partition",),
            ("attention_dispatch", "input_schedule"),
        ),
        (
            {"compile.boundary": "attention_module"},
            ("compile.boundary",),
            ("attention_dispatch", "compile"),
        ),
        (
            {"compile.boundary": "gradient_closure"},
            ("compile.boundary",),
            ("ad_lowering", "compile"),
        ),
        (
            {"fusion.mlp": "torch_inductor"},
            ("fusion.mlp",),
            ("ad_lowering", "fusion"),
        ),
        (
            {"dtensor.params_placement": ("shard",)},
            ("dtensor.params_placement",),
            ("ad_lowering", "distributed_layout"),
        ),
        (
            {"fsdp.mp_policy.reduce_dtype": "bf16"},
            ("fsdp.mp_policy.reduce_dtype",),
            ("distributed_layout", "numeric_backend"),
        ),
        (
            {"inverse_metric.solve_path": "factorized_solve"},
            ("inverse_metric.solve_path",),
            ("inverse_solve", "metric_storage"),
        ),
    ],
)
def test_search_grouping_applies_coupled_axis_rules(
    settings: Mapping[str, object],
    changed_axes: tuple[str, ...],
    groups: tuple[str, ...],
) -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        settings,
        changed_axes=changed_axes,
        admission_status="passed",
    )

    assert (
        run_module._candidate_class_c_groups(
            candidate,
            vpx.axis_table(),
        )
        == groups
    )


def axis_table_candidate(
    settings: Mapping[str, object],
    fixed_fields: Mapping[str, object],
) -> vp.Candidate:
    return vpx.axis_table().admit(
        vp.Candidate("family", "row", settings),
        fixed_fields=fixed_fields,
    )


def assert_axis_table_admitted(
    settings: Mapping[str, object],
    fixed_fields: Mapping[str, object],
) -> None:
    admitted = axis_table_candidate(settings, fixed_fields)

    assert admitted.admission_status == "passed"
    assert admitted.admission_error is None


def assert_axis_table_rejected(
    settings: Mapping[str, object],
    fixed_fields: Mapping[str, object],
    match: str,
) -> None:
    rejected = axis_table_candidate(settings, fixed_fields)

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None
    assert match in rejected.admission_error


def reduction_bound_fields() -> dict[str, object]:
    return {
        "k": 2,
        "epsilon": 0.01,
        "C_op": 1.5,
        "S_row": 3.0,
        "output_norm_floor": 1e-6,
    }


def test_axis_table_admission_requires_exact_loss_scaling_fields() -> None:
    assert_axis_table_admitted(
        {
            "numeric.loss_scaling": "static_scale_with_exact_unscale",
            "numeric.loss_scale": 8.0,
            "numeric.loss_unscale_degree": 1,
        },
        {},
    )
    assert_axis_table_rejected(
        {"numeric.loss_scale": 8.0},
        {},
        "numeric.loss_scaling is required",
    )
    assert_axis_table_rejected(
        {"numeric.loss_scaling": "none", "numeric.loss_scale": 8.0},
        {},
        "forbids",
    )
    assert_axis_table_rejected(
        {"numeric.loss_scaling": "static_scale_with_exact_unscale"},
        {},
        "numeric.loss_scale is required",
    )
    assert_axis_table_rejected(
        {
            "numeric.loss_scaling": "static_scale_with_exact_unscale",
            "numeric.loss_scale": 8.0,
        },
        {},
        "numeric.loss_unscale_degree is required",
    )
    assert_axis_table_rejected(
        {
            "numeric.loss_scaling": "static_scale_with_exact_unscale",
            "numeric.loss_scale": 0.0,
            "numeric.loss_unscale_degree": 1,
        },
        {},
        "positive float",
    )


def test_axis_table_admission_rejects_metric_representation_mismatches() -> None:
    assert_axis_table_admitted(
        {"metric.multiply_path": "dense_matmul"},
        {"metric.representation": "dense_matrix"},
    )
    assert_axis_table_admitted(
        {
            "metric.multiply_path": "factorized_multiply",
            "metric.accumulation": "materialized_blocks",
        },
        {"metric.representation": "kfac_factors"},
    )
    assert_axis_table_admitted(
        {"inverse_metric.solve_path": "woodbury_low_rank_solve"},
        {"metric.representation": "low_rank_factors"},
    )
    assert_axis_table_admitted(
        {"inverse_metric.solve_path": "cholesky_solve"},
        {"metric.representation": "dense_matrix", "metric.psd": True},
    )
    assert_axis_table_admitted(
        {"inverse_metric.solve_path": "eigh_solve"},
        {"metric.representation": "dense_matrix", "metric.symmetric": True},
    )
    assert_axis_table_admitted(
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 4,
            "inverse_metric.preconditioner": "none",
            "metric.multiply_path": "dense_matmul",
        },
        {"metric.representation": "dense_matrix"},
    )
    assert_axis_table_admitted(
        {
            "metric.multiply_path": "blockwise_multiply",
            "metric.block_schedule": "custom_blocks",
            "metric.accumulation": "materialized_blocks",
        },
        {
            "metric.representation": {
                "kind": "block_diagonal",
                "block_schedule": "custom_blocks",
            }
        },
    )
    assert_axis_table_admitted(
        {
            "inverse_metric.solve_path": "blockwise_solve",
            "inverse_metric.block_schedule": "layer_blocks",
        },
        {
            "metric.representation": {
                "kind": "block_diagonal",
                "block_schedule": "layer_blocks",
            }
        },
    )

    assert_axis_table_rejected(
        {"metric.multiply_path": "dense_matmul"},
        {"metric.representation": "kfac_factors"},
        "dense matrix",
    )
    assert_axis_table_rejected(
        {"metric.multiply_path": "factorized_multiply"},
        {"metric.representation": "dense_matrix"},
        "requires factors",
    )
    assert_axis_table_rejected(
        {"metric.multiply_path": "blockwise_multiply"},
        {"metric.representation": "kfac_factors"},
        "requires blocks",
    )
    assert_axis_table_rejected(
        {"inverse_metric.solve_path": "dense_solve"},
        {"metric.representation": "kfac_factors"},
        "requires dense matrix",
    )
    assert_axis_table_rejected(
        {"inverse_metric.solve_path": "cholesky_solve"},
        {"metric.representation": "dense_matrix"},
        "requires a PSD metric",
    )
    assert_axis_table_rejected(
        {"inverse_metric.solve_path": "eigh_solve"},
        {"metric.representation": "dense_matrix"},
        "requires a symmetric metric",
    )
    assert_axis_table_rejected(
        {"inverse_metric.solve_path": "woodbury_low_rank_solve"},
        {"metric.representation": "kfac_factors"},
        "requires low-rank factors",
    )
    assert_axis_table_rejected(
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 4,
            "inverse_metric.preconditioner": "none",
        },
        {"metric.representation": "dense_matrix"},
        "metric.multiply_path",
    )
    assert_axis_table_rejected(
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "metric.multiply_path": "dense_matmul",
        },
        {"metric.representation": "dense_matrix"},
        "iteration_budget",
    )
    assert_axis_table_rejected(
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "metric.multiply_path": "dense_matmul",
            "inverse_metric.iteration_budget": 4,
        },
        {"metric.representation": "dense_matrix"},
        "preconditioner",
    )
    assert_axis_table_rejected(
        {
            "inverse_metric.solve_path": "dense_solve",
            "inverse_metric.preconditioner": "none",
        },
        {"metric.representation": "dense_matrix"},
        "iterative",
    )
    assert_axis_table_rejected(
        {"metric.block_schedule": "layer_blocks"},
        {"metric.representation": "low_rank_factors"},
        "requires blocks or KFAC factors",
    )
    assert_axis_table_rejected(
        {"metric.block_schedule": "layer_blocks"},
        {"metric.representation": "block_diagonal"},
        "representation.block_schedule",
    )
    assert_axis_table_rejected(
        {"metric.block_schedule": "layer_blocks"},
        {
            "metric.representation": {
                "kind": "block_diagonal",
                "block_schedule": "custom_blocks",
            }
        },
        "must match representation.block_schedule",
    )
    assert_axis_table_rejected(
        {"inverse_metric.block_schedule": "module_blocks"},
        {
            "metric.representation": {
                "kind": "kfac_factors",
                "block_schedule": "layer_blocks",
            }
        },
        "must match representation.block_schedule",
    )
    assert_axis_table_rejected(
        {"metric.accumulation": "streaming", "metric.multiply_path": "dense_matmul"},
        {"metric.representation": "dense_matrix"},
        "non-dense metric paths",
    )
    assert_axis_table_rejected(
        {"metric.accumulation": "streaming"},
        {"metric.representation": "kfac_factors"},
        "requires metric.multiply_path",
    )
    assert_axis_table_rejected(
        {"metric.multiply_path": "factorized_multiply"},
        {"metric.representation": "kfac_factors"},
        "metric.accumulation is required",
    )
    assert_axis_table_rejected(
        {
            "metric.multiply_path": "streaming_multiply",
            "metric.accumulation": "materialized_blocks",
        },
        {"metric.representation": "kfac_factors"},
        "must be streaming",
    )
    assert_axis_table_rejected(
        {
            "metric.multiply_path": "factorized_multiply",
            "metric.accumulation": "streaming",
        },
        {"metric.representation": "kfac_factors"},
        "must be materialized_blocks",
    )
    assert_axis_table_rejected(
        {"dtype.metric_factor": "bf16"},
        {"metric.representation": "dense_matrix"},
        "requires a factorized metric path",
    )


def test_axis_table_admission_rejects_cross_axis_contradictions() -> None:
    assert_axis_table_admitted(
        {
            "attention.sdpa_kernel": "math",
            "attention.partition": "packed_tokens",
            "input.batch_layout": "packed_with_inverse_permutation",
        },
        {"attention.calls_sdpa": True},
    )
    assert_axis_table_admitted(
        {
            "compile.enabled": "true",
            "compile.mode": None,
            "compile.options.epilogue_fusion": "true",
            "compile.options.shape_padding": "false",
        },
        {},
    )
    assert_axis_table_admitted(
        {"attention.partition": "segmented_forward_ad", "jvp.path": "forward_ad_dual"},
        {},
    )
    assert_axis_table_admitted(
        {"dtype.accumulation": "bf16"},
        reduction_bound_fields(),
    )
    assert_axis_table_admitted(
        {"sampled_fisher.sample_source": "fixed_seed_and_count"},
        {
            "sampled_fisher.sample_count": 4,
            "sampled_fisher.sample_seed": 123,
        },
    )
    assert_axis_table_admitted(
        {
            "sampled_fisher.sample_source": "fixed_seed_and_count",
            "sampled_fisher.exact_fisher_check": "enabled_with_sampling_bound",
        },
        {
            "sampled_fisher.sample_count": 4,
            "sampled_fisher.sample_seed": 123,
            "sampled_fisher.sampling_bound": {"kind": "absolute_error"},
        },
    )
    assert_axis_table_admitted(
        {"tp.loss_parallel": "true"},
        {
            "tp.exact_cross_shard_normalization": True,
            "tp.multi_rank_agreement_check": True,
        },
    )

    assert_axis_table_rejected(
        {"unknown.axis": "x"},
        {},
        "no axis table owner",
    )
    assert_axis_table_rejected(
        {"vectorization.batch_size": 0},
        {},
        "positive integer",
    )
    assert_axis_table_rejected(
        {"attention.sdpa_kernel": "math"},
        {},
        "calls PyTorch SDPA",
    )
    assert_axis_table_rejected(
        {"attention.sdpa_kernel": "flash_attention"},
        {"attention.calls_sdpa": True, "attention.effective_runtime_dtype": "fp32"},
        "requires float16 or bfloat16",
    )
    assert_axis_table_rejected(
        {"attention.sdpa_kernel": "priority_list"},
        {"attention.calls_sdpa": True},
        "requires backend order",
    )
    assert_axis_table_rejected(
        {"schedule.per_token": "packed", "input.batch_layout": "dense_padded"},
        {},
        "requires packed or variable-length layout",
    )
    assert_axis_table_rejected(
        {
            "activation.recompute": "none",
            "checkpoint.early_stop": "true",
        },
        {},
        "checkpoint.early_stop=false",
    )
    assert_axis_table_rejected(
        {"activation.offload": "saved_tensor_hooks_cpu"},
        {},
        "saved-tensor-hooks path",
    )
    assert_axis_table_rejected(
        {
            "compile.enabled": "false",
            "compile.mode": "max-autotune",
        },
        {},
        "compile.enabled=false forbids",
    )
    assert_axis_table_rejected(
        {
            "compile.mode": "default",
            "compile.options.epilogue_fusion": "true",
        },
        {},
        "backend options require compile.mode=None",
    )
    assert_axis_table_rejected(
        {
            "compile.mode": "default",
            "compile.cuda_graphs": "true",
        },
        {},
        "backend options require compile.mode=None",
    )
    assert_axis_table_rejected(
        {
            "compile.mode": None,
            "compile.options.epilogue_fusion": "false",
            "compile.options.shape_padding": "false",
            "compile.cuda_graphs": "false",
        },
        {},
        "forbid compile.mode=None",
    )
    assert_axis_table_rejected(
        {"numeric.float32_matmul_precision": "medium"},
        {},
        "missing bound fields",
    )
    assert_axis_table_rejected(
        {"sampled_fisher.sample_source": "fixed_sample_table"},
        {"sampled_fisher.sample_count": 4},
        "sample table identity",
    )
    assert_axis_table_rejected(
        {
            "sampled_fisher.sample_source": "fixed_seed_and_count",
            "sampled_fisher.exact_fisher_check": "enabled_with_sampling_bound",
        },
        {
            "sampled_fisher.sample_count": 4,
            "sampled_fisher.sample_seed": 123,
        },
        "sampling-bound formula",
    )
    assert_axis_table_rejected(
        {
            "gradient.value_reuse": "gradient_and_primal_value",
            "gradient.path": "torch_func_grad",
        },
        {},
        "value-and-gradient path",
    )
    assert_axis_table_rejected(
        {
            "jvp.linearize_reuse": "reuse_at_same_primal",
            "jvp.path": "torch_func_jvp",
        },
        {},
        "requires torch_func_linearize",
    )
    assert_axis_table_rejected(
        {
            "vjp.closure_reuse": "reuse_vjp_closure_at_same_primal",
            "vjp.path": "autograd_grad_outputs",
        },
        {},
        "requires torch_func_vjp",
    )
    assert_axis_table_rejected(
        {"ggn.jvp_path": "torch_func_jvp"},
        {},
        "ggn.vjp_path is required",
    )
    assert_axis_table_rejected(
        {"fisher.accumulation": "streaming_dot_accumulate"},
        {},
        "fisher.expectation_path is required",
    )
    assert_axis_table_rejected(
        {
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": "streaming_dot_accumulate",
        },
        {},
        "fisher.score_grad_path is required",
    )
    assert_axis_table_rejected(
        {
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": "materialize_score_gradients",
            "fisher.score_grad_path": "torch_autograd_grad_loop",
        },
        {},
        "not used",
    )
    assert_axis_table_rejected(
        {
            "inverse_metric.solve_path": "dense_solve",
            "inverse_metric.iteration_budget": 4,
        },
        {"metric.representation": "dense_matrix"},
        "applies only to iterative solves",
    )
    assert_axis_table_rejected(
        {"attention.partition": "segmented_forward_ad"},
        {},
        "requires forward AD path",
    )
    assert_axis_table_rejected(
        {"tp.loss_parallel": "true"},
        {"tp.exact_cross_shard_normalization": True},
        "multi-rank agreement check",
    )


def test_cohort_constraint_rejects_unsupported_modes() -> None:
    with pytest.raises(RuntimeError, match="dependency inheritance"):
        vp.CohortConstraint(
            name="bad",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
            dependency_inheritance="all_families",
        )

    with pytest.raises(RuntimeError, match="selection aggregation"):
        vp.CohortConstraint(
            name="bad",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
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
        "candidate_rows": (),
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
    assert vpa.apply_context_parallel
    assert vpa.apply_fsdp2
    assert vpa.apply_fsdp2_group
    assert vpa.apply_tensor_parallel
    assert vpa.DistributedCommunicationBindings
    assert vpa.DistributedContextParallelBindings
    assert vpa.DistributedFSDPBindings
    assert vpa.DistributedMeshBindings
    assert vpa.DistributedPlacementBindings
    assert vpa.DistributedProcessGroupBindings
    assert vpa.DistributedSequenceParallelBindings
    assert vpa.DistributedStrategyBindings
    assert vpa.DistributedTensorParallelBindings
    assert vpa.build_colwise_parallel
    assert vpa.build_device_mesh
    assert vpa.build_dtensor_placement
    assert vpa.build_fsdp_dp_mesh_dims
    assert vpa.build_fsdp_mixed_precision_policy
    assert vpa.build_fsdp_offload_policy
    assert vpa.build_prepare_module_input
    assert vpa.build_prepare_module_output
    assert vpa.build_rowwise_parallel
    assert vpa.build_sequence_parallel
    assert vpa.collective_all_gather_into_tensor
    assert vpa.collective_all_to_all_single
    assert vpa.collective_reduce_scatter_tensor
    assert vpa.initialize_process_group
    assert vpa.named_modules_for_distributed_wrap
    assert vpa.redistribute_dtensor
    assert vpa.resolve_process_group_backend
    assert vpa.run_with_loss_parallel
    assert vpa.wait_collective
    assert vpa.distributed_axis_manifest
    assert vpa.distributed_strategy_axis
    assert vpa.distributed_strategy_applier
    assert vpa.RankCompileTiming
    assert vpa.RankStatus
    assert vpa.PilotReadiness
    assert vpa.check_patched_attention_output_reference
    assert vpa.check_patched_attention_vjp_reference
    assert vpa.load_transformers_model
    assert vpa.register_transformers_attention
    assert vpa.set_transformers_attention_implementation
    assert vpa.transformers_attention_location
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

    grouped = vp.parameter_surface(
        TiedModule(),
        layer_groups=(("first", "second"),),
        block_groups=(("first", "second"),),
    )

    assert grouped.layer_groups == (("first", "second"),)
    assert grouped.block_groups == (("first", "second"),)
    assert grouped.signature()["layer_groups"] == (("first", "second"),)

    with pytest.raises(RuntimeError, match="layer_groups"):
        vp.parameter_surface(TiedModule(), layer_groups=(("first",),))

    with pytest.raises(RuntimeError, match="parametrization_policy"):
        vp.ParameterSurface(
            names=("first",),
            shapes=((2,),),
            trainable=(True,),
            parametrization_policy="disabled",
        )


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
        {"dtype.model_compute": "fp16"},
    )

    assert thresholds["max_abs_diff"] == pytest.approx(1e-4)
    validate_thresholds({"max_abs_diff": 1e-5, "max_rel_diff": 10.0}, thresholds)
    validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 1e-5}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 10.0}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": math.nan}, {"max_abs_diff": 1e-4})

    bound_measurements = numeric_error_bound_measurements(
        {"numeric.float32_matmul_precision": "high"},
        reduction_bound_fields(),
        torch.tensor([4.0]),
    )

    assert bound_measurements["numeric_error_bound_abs"] == pytest.approx(
        1.5 * (0.02 / 0.98) * 3.0
    )
    assert bound_measurements["numeric_error_bound_rel"] == pytest.approx(
        bound_measurements["numeric_error_bound_abs"] / 4.0
    )
    validate_numeric_error_bound(
        {"max_abs_diff": 0.08, "max_rel_diff": 0.02},
        {"max_abs_diff": 0.1, "max_rel_diff": 0.1},
        bound_measurements,
    )

    with pytest.raises(ReferenceFailedError, match="exceeds derived bound"):
        validate_numeric_error_bound(
            {"max_abs_diff": 0.2, "max_rel_diff": 0.2},
            {"max_abs_diff": 0.5, "max_rel_diff": 0.5},
            bound_measurements,
        )

    with pytest.raises(ReferenceFailedError, match="bound exceeds threshold"):
        validate_numeric_error_bound(
            {"max_abs_diff": 0.01, "max_rel_diff": 0.01},
            {"max_abs_diff": 0.01, "max_rel_diff": 0.01},
            bound_measurements,
        )

    with pytest.raises(ReferenceFailedError, match="fields are missing"):
        numeric_error_bound_measurements(
            {"numeric.float32_matmul_precision": "high"},
            {},
            torch.tensor([4.0]),
        )

    bad_bound_fields = dict(reduction_bound_fields())
    bad_bound_fields["k"] = 100

    with pytest.raises(ReferenceFailedError, match=r"k [*] epsilon"):
        numeric_error_bound_measurements(
            {"numeric.float32_matmul_precision": "high"},
            bad_bound_fields,
            torch.tensor([4.0]),
        )

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

    with pytest.raises(ReferenceFailedError):
        assert_tree_close(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0 + 1e-5], dtype=torch.float32),
            settings={"dtype.model_compute": "fp16"},
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


def test_run_candidate_records_compiled_selection_metadata() -> None:
    candidate = vp.Candidate(
        "family",
        "compiled",
        {
            "compile.enabled": "true",
            "compile.cache_state": "cold_compile",
            "compile.compiled_autograd": "false",
            "compile.cuda_graphs": "false",
        },
    )
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    record = run_candidate(
        candidate,
        {"case": "compiled-metadata"},
        operation,
        timing_policy=vp.TimingPolicy(
            short_seconds=10.0,
            short_warmups=0,
            short_measured_calls=2,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 5.0, 5.0, 7.0, 7.0, 9.0)),
    )

    assert record.status == "passed"
    assert record.selection_metadata == {
        "timing_source": "compiled_single_rank",
        "steady_elapsed_seconds": 2.0,
        "compile_time_seconds": 3.0,
        "recompile_count": 0,
        "compile_cache_state": "cold_compile",
        "compile.compiled_autograd": "false",
        "compile.cuda_graphs": "false",
    }


def test_candidate_rows_reject_missing_runtime_bindings() -> None:
    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)
        assert vector is not None

        return reference_passed()

    candidates = (
        vp.Candidate("family", "baseline", {}, admission_status="passed"),
        vp.Candidate(
            "family",
            "fused",
            {"fusion.loss": "fused_ce"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "packed",
            {"input.batch_layout": "packed_with_inverse_permutation"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "lm-head",
            {"chunk.lm_head_weight_chunk_bytes": 1024},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "layer-output",
            {"layout.output": "per_layer_flat"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "block-vector",
            {"layout.vector": "per_block_flat"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "layer-chunk",
            {"chunk.layer_block_size": 2},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "mmap",
            {"memory.vector_residency": "mmap_cpu"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "intermediate",
            {"memory.intermediate_residency": "cpu_staged"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "manual",
            {"activation.recompute": "manual_recompute"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "teacher",
            {"teacher_outputs": "recomputed_with_equality_check"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "stateful",
            {"call.path": "stateful_module"},
            admission_status="passed",
        ),
    )
    runtime = vpx.RuntimeConfig(
        candidates,
        operation_factory,
        reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "fusion_rewriter": False,
            "batch_layout": False,
            "lm_head_chunker": False,
            "mmap_residency": False,
            "intermediate_residency": False,
            "manual_recompute": False,
            "teacher_objective": False,
            "module": True,
            "module_call": None,
            "parameter_surface": {
                "layer_groups": (),
                "block_groups": (),
            },
        },
    )
    rows = {
        candidate.candidate_id: candidate
        for candidate in run_module._candidate_rows(runtime)
    }

    assert rows["baseline"].admission_status == "passed"
    assert rows["fused"].admission_error == (
        "fusion.loss requires a registered fused implementation"
    )
    assert rows["packed"].admission_error == (
        "declared input layout requires a batch_layout binding"
    )
    assert rows["lm-head"].admission_error == (
        "chunk.lm_head_weight_chunk_bytes requires an LM-head chunker binding"
    )
    assert rows["layer-output"].admission_error == (
        "layout.output=per_layer_flat requires declared layer_groups"
    )
    assert rows["block-vector"].admission_error == (
        "layout.vector=per_block_flat requires declared block_groups"
    )
    assert rows["layer-chunk"].admission_error == (
        "chunk.layer_block_size requires declared layer_groups"
    )
    assert rows["mmap"].admission_error == (
        "memory.vector_residency=mmap_cpu requires memory-mapped tensor metadata"
    )
    assert rows["intermediate"].admission_error == (
        "memory.intermediate_residency requires named intermediate boundaries"
    )
    assert rows["manual"].admission_error == (
        "manual_recompute requires recompute-region metadata"
    )
    assert rows["teacher"].admission_error == (
        "recomputed teacher outputs require a teacher objective"
    )
    assert rows["stateful"].admission_error == (
        "stateful_module requires a ModuleCallSpec binding"
    )

    for candidate_id, candidate in rows.items():
        if candidate_id != "baseline":
            assert candidate.admission_status == "failed"


def test_candidate_rows_admit_declared_parameter_groups() -> None:
    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)
        assert vector is not None

        return reference_passed()

    runtime = vpx.RuntimeConfig(
        (
            vp.Candidate(
                "family",
                "grouped",
                {
                    "layout.output": "per_layer_flat",
                    "layout.vector": "per_block_flat",
                    "chunk.layer_block_size": 1,
                },
                admission_status="passed",
            ),
        ),
        operation_factory,
        reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "parameter_surface": {
                "layer_groups": (("w",),),
                "block_groups": (("w",),),
            },
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "passed"


def test_candidate_rows_reject_fusion_without_module() -> None:
    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)
        assert vector is not None

        return reference_passed()

    runtime = vpx.RuntimeConfig(
        (
            vp.Candidate(
                "family",
                "fused",
                {"fusion.loss": "fused_ce"},
                admission_status="passed",
            ),
        ),
        operation_factory,
        reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "fusion_rewriter": True,
            "module": False,
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == "fused rows require a module"


def test_candidate_rows_reject_stateful_module_without_module() -> None:
    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)
        assert vector is not None

        return reference_passed()

    runtime = vpx.RuntimeConfig(
        (
            vp.Candidate(
                "family",
                "stateful",
                {"call.path": "stateful_module"},
                admission_status="passed",
            ),
        ),
        operation_factory,
        reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "module": False,
            "module_call": {"positional_batch_keys": ("scale",)},
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == "call.path=stateful_module requires a module"


def test_candidate_rows_admit_builtin_intermediate_residency_points() -> None:
    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert isinstance(candidate, vp.Candidate)
        assert isinstance(batch, Mapping)
        assert vector is not None

        return reference_passed()

    runtime = vpx.RuntimeConfig(
        (
            vp.Candidate(
                "family",
                "ggn-intermediate",
                {"memory.intermediate_residency": "cpu_staged"},
                admission_status="passed",
            ),
        ),
        operation_factory,
        reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "operator": {"kind": "ggnvp"},
            "intermediate_residency": False,
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "passed"


def test_run_candidate_records_full_size_check_metadata() -> None:
    candidate = vp.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
    )

    def operation() -> torch.Tensor:
        return torch.tensor([1.0])

    def full_size_check(
        output: vp.TensorTree,
        samples: tuple[vp.Measurement, ...],
    ) -> Mapping[str, object]:
        assert isinstance(output, torch.Tensor)
        assert len(samples) == 1
        torch.testing.assert_close(output, torch.tensor([1.0]))

        return {
            "full_size_agreement_passed": True,
            "full_size_agreement_name": "tests.full_size_gate",
        }

    record = run_candidate(
        candidate,
        {"case": "full-size-check"},
        operation,
        timing_policy=vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
        full_size_check=full_size_check,
    )

    assert record.status == "passed"
    assert record.selection_metadata["full_size_agreement_passed"] is True
    assert record.selection_metadata["full_size_agreement_name"] == (
        "tests.full_size_gate"
    )


def test_run_candidate_records_measured_recompile_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = vp.Candidate(
        "family",
        "compiled",
        {
            "compile.enabled": "true",
            "compile.cache_state": "cold_compile",
            "compile.compiled_autograd": "false",
            "compile.cuda_graphs": "false",
        },
    )
    counters = {"stats": {"unique_graphs": 10}}
    calls = {"count": 0}

    fake_dynamo_utils = types.SimpleNamespace(counters=counters)

    def fake_import_module(name: str) -> object:
        assert name == "torch._dynamo.utils"

        return fake_dynamo_utils

    def operation() -> torch.Tensor:
        calls["count"] += 1

        if calls["count"] == 1:
            counters["stats"]["unique_graphs"] += 3

        return torch.tensor(float(calls["count"]))

    monkeypatch.setattr(measure_module.importlib, "import_module", fake_import_module)
    record = run_candidate(
        candidate,
        {"case": "compiled-recompile-count"},
        operation,
        timing_policy=vp.TimingPolicy(
            short_seconds=10.0,
            short_warmups=0,
            short_measured_calls=2,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 5.0, 5.0, 7.0, 7.0, 9.0)),
    )

    assert record.selection_metadata["recompile_count"] == 2


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

    with pytest.raises(vp.MeasurementError) as error_info:
        measure_once(operation, memory_backend=backend)

    error = error_info.value

    assert isinstance(error, measure_module.OperationMeasurementError)
    assert error.error_type == "RuntimeError"
    assert error.samples[0].device == "cpu"
    assert backend.cleanup_calls == 2


def _record(
    candidate: vp.Candidate,
    *,
    elapsed: tuple[float, ...],
    reserved: tuple[float, ...],
    input_signature: Mapping[str, object],
    status: str = "passed",
    reference_passed: bool = True,
    selection_metadata: Mapping[str, object] | None = None,
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
        selection_metadata={}
        if selection_metadata is None
        else dict(selection_metadata),
        dependency_identities=dict(candidate.dependency_identities),
        reference_passed=reference_passed,
    )

    return record


def _with_rank_memory_samples(
    record: FullSizeRecord,
    reserved: tuple[float, ...],
) -> FullSizeRecord:
    samples = tuple(
        Measurement(
            elapsed_seconds=record.timing_samples[0].elapsed_seconds,
            peak_allocated_mib=memory,
            peak_reserved_mib=memory,
            post_allocated_mib=0.0,
            post_reserved_mib=0.0,
            rank=rank,
            device=f"cuda:{rank}",
        )
        for rank, memory in enumerate(reserved)
    )

    return dataclasses.replace(record, memory_samples=samples)


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


@pytest.mark.parametrize(
    "settings",
    [
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
        },
        {"attention.frontend": "transformers_flex_attention"},
        {
            "attention.frontend": "packed_exact",
            "attention.partition": "packed_tokens",
        },
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
        },
        {"compile.cuda_graphs": "true"},
        {"compile.mode": "max-autotune"},
        {"fusion.loss": "fused_ce"},
        {"tp.loss_parallel": "true"},
        {"context_parallel.enabled": "true"},
    ],
)
def test_selection_requires_full_size_agreement(settings: Mapping[str, object]) -> None:
    signature = {"case": "full-size-gate", "settings": dict(settings)}
    policy = vp.SelectionPolicy()
    gated = vp.Candidate(
        "family",
        "gated",
        settings,
    )
    fallback = vp.Candidate("family", "fallback", {})

    selected, _ = select_family(
        (
            (
                gated,
                _record(
                    gated,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
            (
                fallback,
                _record(
                    fallback,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected == fallback

    selected_with_gate, _ = select_family(
        (
            (
                gated,
                _record(
                    gated,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                    selection_metadata={"full_size_agreement_passed": True},
                ),
            ),
            (
                fallback,
                _record(
                    fallback,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected_with_gate == gated


def test_selection_scores_compiled_rows_by_call_horizon() -> None:
    signature = {"case": "compiled-selection"}
    eager = vp.Candidate("family", "eager", {})
    compiled = vp.Candidate(
        "family",
        "compiled",
        {"compile.enabled": "true"},
    )
    eager_record = _record(
        eager,
        elapsed=(4.0,),
        reserved=(1.0,),
        input_signature=signature,
    )
    compiled_record = _record(
        compiled,
        elapsed=(2.0,),
        reserved=(1.0,),
        input_signature=signature,
        selection_metadata={
            "steady_elapsed_seconds": 2.0,
            "compile_time_seconds": 30.0,
            "recompile_count": 0,
        },
    )
    short_horizon = vp.SelectionPolicy(compile_call_horizon=10)
    long_horizon = vp.SelectionPolicy(compile_call_horizon=30)

    short_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=short_horizon,
    )
    long_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=long_horizon,
    )

    assert short_selected == eager
    assert long_selected == compiled


def test_selection_scores_distributed_rows_by_global_elapsed_seconds() -> None:
    signature = {"case": "distributed-selection"}
    local = vp.Candidate("family", "local", {})
    distributed = vp.Candidate(
        "family",
        "distributed",
        {"distributed.strategy": "fsdp2"},
    )
    selected, _ = select_family(
        (
            (
                local,
                _record(
                    local,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
            (
                distributed,
                _record(
                    distributed,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                    selection_metadata={"global_elapsed_seconds": 3.0},
                ),
            ),
        ),
        input_signature=signature,
        policy=vp.SelectionPolicy(),
    )

    assert selected == local

    with pytest.raises(TypeError, match="global_elapsed_seconds"):
        select_family(
            (
                (
                    distributed,
                    _record(
                        distributed,
                        elapsed=(1.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            ),
            input_signature=signature,
            policy=vp.SelectionPolicy(),
        )


def test_selection_scores_compiled_distributed_rows_by_global_compile_fields() -> None:
    signature = {"case": "compiled-distributed-selection"}
    eager = vp.Candidate("family", "eager", {})
    compiled = vp.Candidate(
        "family",
        "compiled",
        {"compile.enabled": "true", "distributed.strategy": "fsdp2"},
    )
    eager_record = _record(
        eager,
        elapsed=(4.0,),
        reserved=(1.0,),
        input_signature=signature,
    )
    compiled_record = _record(
        compiled,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=signature,
        selection_metadata={
            "global_steady_elapsed_seconds": 1.0,
            "global_compile_time_seconds": 90.0,
            "recompile_count": 0,
        },
    )
    short_horizon = vp.SelectionPolicy(compile_call_horizon=10)
    long_horizon = vp.SelectionPolicy(compile_call_horizon=90)

    short_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=short_horizon,
    )
    long_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=long_horizon,
    )

    assert short_selected == eager
    assert long_selected == compiled


def test_selection_tie_breaks_with_declared_rank_memory_reduction() -> None:
    signature = {"case": "distributed-memory-selection"}
    first = vp.Candidate("family", "first", {"distributed.strategy": "fsdp2"})
    second = vp.Candidate("family", "second", {"distributed.strategy": "fsdp2"})
    first_record = _with_rank_memory_samples(
        _record(
            first,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
            selection_metadata={"global_elapsed_seconds": 1.0},
        ),
        (60.0, 1.0),
    )
    second_record = _with_rank_memory_samples(
        _record(
            second,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
            selection_metadata={"global_elapsed_seconds": 1.0},
        ),
        (40.0, 40.0),
    )
    max_selected, _ = select_family(
        ((first, first_record), (second, second_record)),
        input_signature=signature,
        policy=vp.SelectionPolicy(rank_memory_reduction="max_peak_reserved"),
    )
    sum_selected, _ = select_family(
        ((first, first_record), (second, second_record)),
        input_signature=signature,
        policy=vp.SelectionPolicy(rank_memory_reduction="sum_peak_reserved"),
    )

    assert max_selected == second
    assert sum_selected == first


def test_cohort_selection_sums_compiled_row_scores() -> None:
    signature = {"case": "compiled-cohort"}
    eager_a = vp.Candidate("a", "eager-a", {})
    eager_b = vp.Candidate("b", "eager-b", {})
    compiled_a = vp.Candidate("a", "compiled-a", {"compile.enabled": "true"})
    compiled_b = vp.Candidate("b", "compiled-b", {"compile.enabled": "true"})
    policy = vp.SelectionPolicy(compile_call_horizon=30)
    cohort = select_cohort(
        (
            {
                "a": (
                    eager_a,
                    _record(
                        eager_a,
                        elapsed=(4.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    eager_b,
                    _record(
                        eager_b,
                        elapsed=(4.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            },
            {
                "a": (
                    compiled_a,
                    _record(
                        compiled_a,
                        elapsed=(2.0,),
                        reserved=(2.0,),
                        input_signature=signature,
                        selection_metadata={
                            "steady_elapsed_seconds": 2.0,
                            "compile_time_seconds": 30.0,
                            "recompile_count": 0,
                        },
                    ),
                ),
                "b": (
                    compiled_b,
                    _record(
                        compiled_b,
                        elapsed=(2.0,),
                        reserved=(2.0,),
                        input_signature=signature,
                        selection_metadata={
                            "steady_elapsed_seconds": 2.0,
                            "compile_time_seconds": 30.0,
                            "recompile_count": 0,
                        },
                    ),
                ),
            },
        ),
        families=("a", "b"),
        policy=policy,
    )

    assert cohort["a"][0] == compiled_a
    assert cohort["b"][0] == compiled_b


def test_selection_rejects_unsupported_policy_fields() -> None:
    candidate = vp.Candidate("family", "row", {})
    policies = (
        vp.SelectionPolicy(speed_statistic="mean_elapsed_seconds"),
        vp.SelectionPolicy(compiled_speed_statistic="steady_elapsed_seconds"),
        vp.SelectionPolicy(distributed_speed_statistic="rank_zero_elapsed_seconds"),
        vp.SelectionPolicy(rank_memory_reduction="rank_zero_peak_reserved"),
        vp.SelectionPolicy(cohort_speed_statistic="sum_selection_score_seconds"),
        vp.SelectionPolicy(accepted_status="passed_only"),
    )

    for policy in policies:
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
                policy=policy,
            )


def test_cohort_constraint_uses_spec_selection_aggregation_token() -> None:
    constraint = vp.CohortConstraint(
        name="dtype",
        settings_keys=("dtype.model_compute",),
        assignments=({"dtype.model_compute": "fp32"},),
    )

    assert constraint.selection_aggregation == "sum_median_elapsed_seconds"

    with pytest.raises(RuntimeError):
        vp.CohortConstraint(
            name="old-token",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
            selection_aggregation="sum_selection_score_seconds",
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


def test_record_current_rejects_stale_candidate_and_full_size_status() -> None:
    input_signature = _input_signature("status-current")
    candidate = vp.Candidate(
        "family",
        "row",
        {"dtype": "fp32"},
        admission_status="passed",
    )
    candidate_row = vpx.candidate_record_to_json(candidate, input_signature)

    assert record_current(
        candidate_row,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert not record_current(
        candidate_row,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="failed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )

    full_size_row = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    full_size_json = vpx.full_size_record_to_json(full_size_row)

    assert full_size_row.row_key()["status"] == "passed"
    assert record_current(
        full_size_json,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert not record_current(
        full_size_json,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="failed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )


def test_reference_row_key_includes_row_and_check_identity() -> None:
    candidate_a = vp.Candidate("family", "row-a", {"axis": "same"})
    candidate_b = vp.Candidate("family", "row-b", {"axis": "same"})
    first = _check_record(candidate_a, input_signature={})
    second = _check_record(candidate_b, input_signature={})
    third = dataclasses.replace(first, name="second")
    fourth = dataclasses.replace(first, status="failed")
    fifth = dataclasses.replace(first, thresholds={"max_abs_diff": 1e-3})

    assert first.row_key() != second.row_key()
    assert first.row_key() != third.row_key()
    assert first.row_key() != fourth.row_key()
    assert first.row_key() != fifth.row_key()


def test_check_record_current_rejects_stale_status_and_thresholds() -> None:
    record = _check_record(vp.Candidate("family", "row", {}), input_signature={})

    assert not vpx.check_record_current(dataclasses.replace(record, status="failed"))
    assert not vpx.check_record_current(
        dataclasses.replace(record, thresholds={"max_abs_diff": 1e-3})
    )


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
            "vectorization.randomness": "error",
            "requires_forward_ad": False,
            "forward_ad_supported": False,
        })

    with pytest.raises(vp.AdmissionError):
        vpx.admit_checkpoint({
            "checkpoint.use_reentrant": "true",
            "checkpoint.preserve_rng_state": "true",
            "checkpoint.determinism_check": "default",
            "checkpoint.context_fn": "none",
            "checkpoint.early_stop": "true",
            "checkpoint.moves_to_new_device": "false",
            "checkpoint.uses_global_state": "false",
        })


def checkpoint_fields() -> dict[str, object]:
    return {
        "activation.offload": "none",
        "checkpoint.use_reentrant": "false",
        "checkpoint.preserve_rng_state": "true",
        "checkpoint.determinism_check": "default",
        "checkpoint.context_fn": "none",
        "checkpoint.early_stop": "true",
        "checkpoint.moves_to_new_device": "false",
        "checkpoint.uses_global_state": "false",
    }


def test_checkpoint_operation_preserves_rng_state() -> None:
    values = []
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
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
        policy_key="activation.recompute",
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
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.preserve_rng_state": "false",
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
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert len(values) == 2
    assert not torch.equal(values[0], values[1])


def test_checkpoint_operation_uses_declared_context_pair() -> None:
    events = []
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.context_fn": "declared_context_pair",
            "checkpoint.context_fn_callable": lambda: (
                contextlib.nullcontext(),
                contextlib.nullcontext(),
            ),
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        events.append("called")

        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert events == ["called", "called"]


def test_checkpoint_operation_executes_selective_checkpoint_context_pair() -> None:
    events = []
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_selective",
            **checkpoint_fields(),
            "checkpoint.context_fn": "declared_context_pair",
            "checkpoint.context_fn_callable": lambda: (
                contextlib.nullcontext(),
                contextlib.nullcontext(),
            ),
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        events.append("called")

        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert events == ["called", "called"]


def test_checkpoint_operation_rejects_declared_context_without_callable() -> None:
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.context_fn": "declared_context_pair",
        },
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square()

    with pytest.raises(vp.AdmissionError, match="requires callable"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="activation.recompute",
        )()


def test_checkpoint_operation_rejects_unadmitted_fields_before_execution() -> None:
    calls = []
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.use_reentrant": "true",
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
            policy_key="activation.recompute",
        )()

    assert calls == []


def test_checkpoint_operation_runs_cpu_saved_tensor_hooks() -> None:
    vector = torch.tensor([2.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "saved_tensor_hooks_cpu",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert torch.equal(vector.grad, torch.tensor([4.0]))


def test_checkpoint_operation_runs_custom_saved_tensor_hooks() -> None:
    events = []
    vector = torch.tensor([2.0], requires_grad=True)

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("pack", tensor.detach().clone()))

        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("unpack", tensor.detach().clone()))

        return tensor

    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "custom_saved_tensor_hooks",
            "activation.pack_hook": pack_hook,
            "activation.unpack_hook": unpack_hook,
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert tuple(event for event, _ in events) == ("pack", "unpack")
    assert vector.grad is not None
    assert torch.equal(vector.grad, torch.tensor([4.0]))


def test_checkpoint_operation_rejects_missing_activation_offload() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"activation.recompute": "none"},
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(vp.AdmissionError, match=r"activation[.]offload"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (torch.tensor([1.0]),),
            policy_key="activation.recompute",
        )


def test_checkpoint_operation_rejects_custom_offload_without_hooks() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "custom_saved_tensor_hooks",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(vp.AdmissionError, match="pack and unpack"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (torch.tensor([1.0]),),
            policy_key="activation.recompute",
        )()


def test_adapter_runtime_executes_checkpoint_without_standard_runtime_support(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    settings = {
        "operator_path": "autograd_grad",
        "activation.recompute": "checkpoint_non_reentrant_by_layer",
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
            policy_key="activation.recompute",
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
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_flash = vp.Candidate(
        "family",
        "float-flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "fp32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash3 = vp.Candidate(
        "family",
        "flash3",
        {
            "attention.frontend": "transformers_flash_attention_3",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash4 = vp.Candidate(
        "family",
        "flash4",
        {
            "attention.frontend": "transformers_flash_attention_4",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    paged_flash = vp.Candidate(
        "family",
        "paged-flash",
        {
            "attention.frontend": "paged|flash_attention_4",
            "dtype.model_compute": "bf16",
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
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    math_attentions = vp.Candidate(
        "family",
        "math-attentions",
        {
            "attention.frontend": "transformers_sdpa",
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
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "flash_attention",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_direct_flash_kernel = vp.Candidate(
        "family",
        "float-sdpa-flash",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "flash_attention",
            "dtype.model_compute": "fp32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    priority_sdpa = vp.Candidate(
        "family",
        "priority-sdpa",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    bad_priority_sdpa = vp.Candidate(
        "family",
        "bad-priority-sdpa",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("priority_list",),
            "dtype.model_compute": "bf16",
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
        if candidate.settings["dtype"] == "fp16":
            return False, "float16 disabled"

        return True, None

    registry.register(
        vpx.AxisDescriptor(
            "dtype",
            ("dtype",),
            ("fp32", "fp16"),
            admission_rule=admit_dtype,
        )
    )

    with pytest.raises(vp.AdmissionError):
        registry.register(vpx.AxisDescriptor("other", ("dtype",), ("float64",)))

    candidates = vpx.settings_product(
        "family",
        {"dtype": ("fp32", "fp16")},
        generator_id="grid",
        generator_version="1",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.settings["dtype"] == "fp32"
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
        assert candidate.settings["dtype"] == "fp32"
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
    assert plan.selected["family"].settings == {"dtype": "fp32"}
    assert plan.full_size_records[1].status == "failed"
    assert plan.full_size_records[1].error_type == "AdmissionError"

    invalid = vp.Candidate(
        "family",
        "invalid",
        {"dtype": "float64"},
        changed_axes=("dtype",),
    )

    assert registry.admit(invalid).admission_status == "failed"


def test_tune_records_runtime_full_size_check_metadata() -> None:
    class PassingFullSizeCheck:
        def __init__(self) -> None:
            self.calls = []

        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"full_size_check": "tests.passing"}

        def __call__(
            self,
            candidate: vp.Candidate,
            inputs: tuple[tuple[vp.Batch, vp.TensorTree], ...],
            output: vp.TensorTree,
            samples: tuple[vp.Measurement, ...],
        ) -> Mapping[str, object]:
            assert candidate.candidate_id == "flash"
            assert len(inputs) == 1
            assert isinstance(output, tuple)
            assert len(samples) == 1
            self.calls.append(candidate.candidate_id)

            return {"full_size_agreement_passed": True}

    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
        admission_status="passed",
    )
    full_size_check = PassingFullSizeCheck()

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "flash"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "flash"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=dataclasses.replace(
            cpu_target(
                vp.TimingPolicy(
                    short_seconds=0.0,
                    medium_seconds=0.0,
                    long_warmups=0,
                    long_measured_calls=1,
                )
            ),
            allowed_attention_frontends=("pytorch_sdpa_direct",),
            allowed_sdpa_kernels=("flash_attention",),
        ),
        runtime=vpx.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "full-size-check"},
            full_size_check=full_size_check,
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert full_size_check.calls == ["flash"]
    assert problem.runtime.identity()["full_size_check"] == {
        "full_size_check": "tests.passing"
    }
    assert plan.selected["family"].candidate_id == "flash"
    assert (
        plan.records["family"].selection_metadata["full_size_agreement_passed"] is True
    )


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
        "vectorization.randomness": "error",
        "requires_forward_ad": True,
        "forward_ad_supported": True,
    }
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "dtype.model_compute": "bf16",
            "hvp.path": "reverse_over_reverse",
            "numeric.float32_matmul_precision": "high",
        },
    )
    bad_flag = vp.Candidate(
        "family",
        "bad-flag",
        {"numeric.bf16_reduced_precision_reduction": True},
    )
    bad_dtype = vp.Candidate("family", "bad-dtype", {"dtype.model_compute": "float64"})
    fp8_storage = vp.Candidate(
        "family",
        "fp8-storage",
        {"dtype.parameter_storage": "fp8_when_supported"},
    )
    fp8_compute = vp.Candidate(
        "family",
        "fp8-compute",
        {"dtype.model_compute": "fp8_when_supported"},
    )
    bad_path = vp.Candidate(
        "family",
        "bad-path",
        {"hvp.path": "reverse_over_forward"},
    )
    missing_torch_func_fields = vp.Candidate(
        "family",
        "missing-torch-func-fields",
        {"jvp.path": "torch_func_jvp"},
    )
    valid_torch_func = vp.Candidate(
        "family",
        "valid-torch-func",
        {"jvp.path": "torch_func_jvp", **torch_func_fields},
    )
    valid_ggn_linearize = vp.Candidate(
        "family",
        "valid-ggn-linearize",
        {
            "ggn.jvp_path": "torch_func_linearize",
            "ggn.vjp_path": "torch_func_vjp",
            **torch_func_fields,
        },
    )
    missing_ggn_linearize_fields = vp.Candidate(
        "family",
        "missing-ggn-linearize-fields",
        {
            "ggn.jvp_path": "torch_func_linearize",
            "ggn.vjp_path": "torch_func_vjp",
        },
    )
    valid_forward_ad = vp.Candidate(
        "family",
        "valid-forward-ad",
        {
            "jvp.path": "forward_ad_dual",
            "requires_forward_ad": True,
            "forward_ad_supported": True,
        },
    )
    unsupported_forward_ad = vp.Candidate(
        "family",
        "unsupported-forward-ad",
        {
            "jvp.path": "forward_ad_dual",
            "requires_forward_ad": True,
            "forward_ad_supported": False,
        },
    )
    valid_vmap = vp.Candidate(
        "family",
        "valid-vmap",
        {
            "empirical_fisher.grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "schedule.per_example": "vmap",
            "batch.empirical_example_batch_size": 2,
        },
    )
    valid_sampled_vmap = vp.Candidate(
        "family",
        "valid-sampled-vmap",
        {
            "sampled_fisher.score_grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "schedule.per_example": "vmap",
            "batch.fisher_sample_batch_size": 2,
        },
    )
    valid_hvp_vmap = vp.Candidate(
        "family",
        "valid-hvp-vmap",
        {
            "hvp.path": "linearize_grad",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    valid_jvp_vmap = vp.Candidate(
        "family",
        "valid-jvp-vmap",
        {
            "jvp.path": "torch_func_jvp",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    valid_vjp_vmap = vp.Candidate(
        "family",
        "valid-vjp-vmap",
        {
            "vjp.path": "torch_func_vjp",
            **torch_func_fields,
            "requires_forward_ad": False,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"y": 0},
        },
    )
    valid_ggn_vmap = vp.Candidate(
        "family",
        "valid-ggn-vmap",
        {
            "ggn.jvp_path": "torch_func_jvp",
            "ggn.vjp_path": "torch_func_vjp",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    valid_fisher_vector_vmap = vp.Candidate(
        "family",
        "valid-fisher-vector-vmap",
        {
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": "materialize_score_gradients",
            **torch_func_fields,
            "requires_forward_ad": False,
            "forward_ad_supported": False,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
            "vectorization.randomness": "error",
        },
    )
    valid_composition_vmap = vp.Candidate(
        "family",
        "valid-composition-vmap",
        {
            "composition.execution": "stream_child_outputs",
            **torch_func_fields,
            "requires_forward_ad": False,
            "forward_ad_supported": False,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
            "vectorization.randomness": "error",
        },
    )
    missing_vmap_randomness = vp.Candidate(
        "family",
        "missing-vmap-randomness",
        {
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": "materialize_score_gradients",
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    valid_manual_batch = vp.Candidate(
        "family",
        "valid-manual-batch",
        {
            "hvp.path": "reverse_over_reverse",
            "vectorization.mode": "manual_batch",
            "vectorization.batch_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    rejected_ggn_vmap_autograd_vjp = vp.Candidate(
        "family",
        "rejected-ggn-vmap-autograd-vjp",
        {
            "ggn.jvp_path": "torch_func_jvp",
            "ggn.vjp_path": "autograd_grad_outputs",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    rejected_forward_ad_jvp_vmap = vp.Candidate(
        "family",
        "rejected-forward-ad-jvp-vmap",
        {
            "jvp.path": "forward_ad_dual",
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"w": 0},
        },
    )
    valid_hvp_single_loop = vp.Candidate(
        "family",
        "valid-hvp-single-loop",
        {
            "hvp.path": "reverse_over_reverse",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
    )
    missing_sampled_vmap_schedule = vp.Candidate(
        "family",
        "missing-sampled-vmap-schedule",
        {
            "sampled_fisher.score_grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "batch.fisher_sample_batch_size": 2,
        },
    )
    stray_vmap_in_dims = vp.Candidate(
        "family",
        "stray-vmap-in-dims",
        {
            "empirical_fisher.grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "schedule.per_example": "vmap",
            "batch.empirical_example_batch_size": 2,
            "vectorization.in_dims": {"x": 0},
        },
    )
    invalid_vmap_in_dims = vp.Candidate(
        "family",
        "invalid-vmap-in-dims",
        {
            "hvp.path": "linearize_grad",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": {"x": "0"},
        },
    )
    forward_ad_vmap = vp.Candidate(
        "family",
        "forward-ad-vmap",
        {
            "empirical_fisher.grad_path": "vmap_grad",
            **torch_func_fields,
            "schedule.per_example": "vmap",
            "batch.empirical_example_batch_size": 2,
        },
    )
    missing_vmap_schedule = vp.Candidate(
        "family",
        "missing-vmap-schedule",
        {
            "empirical_fisher.grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "batch.empirical_example_batch_size": 2,
        },
    )
    valid_manual_per_example = vp.Candidate(
        "family",
        "valid-manual-per-example",
        {
            "fisher.score_grad_path": "torch_autograd_grad_loop",
            "schedule.per_example": "manual_batch",
            "batch.fisher_sample_batch_size": 2,
        },
    )
    missing_manual_per_example_size = vp.Candidate(
        "family",
        "missing-manual-per-example-size",
        {
            "sampled_fisher.score_grad_path": "torch_autograd_grad_loop",
            "schedule.per_example": "manual_batch",
        },
    )
    single_loop_vmap = vp.Candidate(
        "family",
        "single-loop-vmap",
        {
            "empirical_fisher.grad_path": "vmap_grad",
            **torch_func_fields,
            "requires_forward_ad": False,
            "schedule.per_example": "vmap",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"x": 0, "normalization": None},
        },
    )
    vmap_without_vmap_path = vp.Candidate(
        "family",
        "vmap-without-vmap-path",
        {
            "empirical_fisher.grad_path": "torch_autograd_grad_loop",
            "vectorization.mode": "vmap",
        },
    )
    manual_batch = vp.Candidate(
        "family",
        "manual-batch",
        {"vectorization.mode": "manual_batch"},
    )
    valid_microbatch_accumulation = vp.Candidate(
        "family",
        "valid-microbatch-accumulation",
        {
            "gradient.path": "torch_autograd_grad",
            "schedule.gradient_accumulation": "microbatch_accumulate",
            "batch.data_microbatch_size": 2,
        },
    )
    missing_microbatch_size = vp.Candidate(
        "family",
        "missing-microbatch-size",
        {
            "gradient.path": "torch_autograd_grad",
            "schedule.gradient_accumulation": "microbatch_accumulate",
        },
    )
    stray_microbatch_size = vp.Candidate(
        "family",
        "stray-microbatch-size",
        {"batch.data_microbatch_size": 2},
    )
    jvp_microbatch = vp.Candidate(
        "family",
        "jvp-microbatch",
        {
            "jvp.path": "torch_func_jvp",
            **torch_func_fields,
            "requires_forward_ad": True,
            "forward_ad_supported": True,
            "schedule.gradient_accumulation": "microbatch_accumulate",
            "batch.data_microbatch_size": 2,
        },
    )
    hvp_microbatch = vp.Candidate(
        "family",
        "hvp-microbatch",
        {
            "hvp.path": "reverse_over_reverse",
            "schedule.gradient_accumulation": "microbatch_accumulate",
            "batch.data_microbatch_size": 2,
        },
    )
    valid_input_memory_axes = vp.Candidate(
        "family",
        "valid-input-memory-axes",
        {
            "schedule.per_token": "loop",
            "input.batch_layout": "dense_padded",
            "input.length_grouping": "none",
            "input.host_to_device": "outside_measured_call",
            "input.residency": "cpu_staged",
            "teacher_outputs": "precomputed_cpu",
            "memory.vector_residency": "cpu_staged",
            "memory.intermediate_residency": "cpu_staged",
            "memory.factor_residency": "cpu_staged",
            "memory.output_buffers": "fresh_allocation",
        },
    )
    valid_package_runtime_axes = vp.Candidate(
        "family",
        "valid-package-runtime-axes",
        {
            "dtype.autodiff_compute": "bf16",
            "dtype.accumulation": "fp32",
            "gradient.graph_schedule": "build_once",
            "memory.primal_outputs": "retain",
            "memory.jvp_outputs": "retain",
            "memory.output_cotangents": "retain",
            "activation.recompute": "checkpoint_selective",
            "activation.offload": "custom_saved_tensor_hooks",
            "activation.pack_hook": lambda tensor: tensor,
            "activation.unpack_hook": lambda tensor: tensor,
            "checkpoint.use_reentrant": "false",
            "checkpoint.early_stop": "true",
            "checkpoint.preserve_rng_state": "false",
            "checkpoint.determinism_check": "default",
            "checkpoint.context_fn": "declared_context_pair",
            "checkpoint.context_fn_callable": lambda: (
                contextlib.nullcontext(),
                contextlib.nullcontext(),
            ),
            "checkpoint.moves_to_new_device": "false",
            "checkpoint.uses_global_state": "false",
            "metric.block_schedule": "layer_blocks",
            "inverse_metric.block_schedule": "module_blocks",
            "fusion.norm": "model_default",
            "fusion.mlp": "model_default",
            "fusion.rope": "model_default",
            "fusion.logits": "model_default",
            "fusion.loss": "model_default",
        },
    )
    invalid_package_runtime_axis = vp.Candidate(
        "family",
        "invalid-package-runtime-axis",
        {"checkpoint.use_reentrant": "true"},
    )
    valid_compile_boundary = vp.Candidate(
        "family",
        "valid-compile-boundary",
        {"compile.boundary": "metric_multiply"},
    )
    invalid_compile_boundary = vp.Candidate(
        "family",
        "invalid-compile-boundary",
        {"compile.boundary": "unknown_boundary"},
    )
    valid_chunk_axes = vp.Candidate(
        "family",
        "valid-chunk-axes",
        {
            "batch.hvp_row_batch_size": 2,
            "batch.ggn_batch_size": 3,
            "chunk.token_block_size": 4,
            "chunk.output_cotangent_block_size": 6,
            "chunk.parameter_block_size": 7,
            "chunk.layer_block_size": 8,
            "chunk.lm_head_weight_chunk_bytes": 9,
        },
    )
    invalid_chunk_axis = vp.Candidate(
        "family",
        "invalid-chunk-axis",
        {"chunk.token_block_size": 0},
    )

    assert registry.admit(candidate).admission_status == "passed"
    assert registry.admit(bad_flag).admission_status == "failed"
    assert registry.admit(bad_dtype).admission_status == "failed"
    assert registry.admit(bad_path).admission_status == "failed"
    assert registry.admit(missing_torch_func_fields).admission_status == "failed"
    assert registry.admit(valid_torch_func).admission_status == "passed"
    assert registry.admit(valid_ggn_linearize).admission_status == "passed"
    assert registry.admit(missing_ggn_linearize_fields).admission_status == "failed"
    assert registry.admit(valid_forward_ad).admission_status == "passed"
    assert registry.admit(unsupported_forward_ad).admission_status == "failed"
    assert registry.admit(valid_vmap).admission_status == "passed"
    assert registry.admit(valid_sampled_vmap).admission_status == "passed"
    assert registry.admit(valid_hvp_vmap).admission_status == "passed"
    assert registry.admit(valid_jvp_vmap).admission_status == "passed"
    assert registry.admit(valid_vjp_vmap).admission_status == "passed"
    assert registry.admit(valid_ggn_vmap).admission_status == "passed"
    assert registry.admit(valid_fisher_vector_vmap).admission_status == "passed"
    assert registry.admit(valid_composition_vmap).admission_status == "passed"
    assert registry.admit(missing_vmap_randomness).admission_status == "failed"
    assert registry.admit(valid_manual_batch).admission_status == "passed"
    assert registry.admit(rejected_ggn_vmap_autograd_vjp).admission_status == "failed"
    assert registry.admit(rejected_forward_ad_jvp_vmap).admission_status == "failed"
    assert registry.admit(valid_hvp_single_loop).admission_status == "passed"
    assert registry.admit(missing_sampled_vmap_schedule).admission_status == "failed"
    assert registry.admit(stray_vmap_in_dims).admission_status == "failed"
    assert registry.admit(invalid_vmap_in_dims).admission_status == "failed"
    assert registry.admit(forward_ad_vmap).admission_status == "failed"
    assert registry.admit(missing_vmap_schedule).admission_status == "failed"
    assert registry.admit(valid_manual_per_example).admission_status == "passed"
    assert registry.admit(missing_manual_per_example_size).admission_status == "failed"
    assert registry.admit(single_loop_vmap).admission_status == "passed"
    assert registry.admit(vmap_without_vmap_path).admission_status == "failed"
    assert registry.admit(manual_batch).admission_status == "failed"
    assert registry.admit(valid_microbatch_accumulation).admission_status == "passed"
    assert registry.admit(missing_microbatch_size).admission_status == "failed"
    assert registry.admit(stray_microbatch_size).admission_status == "failed"
    assert registry.admit(jvp_microbatch).admission_status == "passed"
    assert registry.admit(hvp_microbatch).admission_status == "passed"
    assert registry.admit(valid_input_memory_axes).admission_status == "passed"
    assert registry.admit(valid_package_runtime_axes).admission_status == "passed"
    assert registry.admit(invalid_package_runtime_axis).admission_status == "failed"
    assert registry.admit(valid_compile_boundary).admission_status == "passed"
    assert registry.admit(invalid_compile_boundary).admission_status == "failed"
    assert registry.admit(valid_chunk_axes).admission_status == "passed"
    assert registry.admit(invalid_chunk_axis).admission_status == "failed"
    assert registry.axes["dtype.model_compute"].admit(bad_dtype)[0] is False
    assert registry.admit(fp8_storage).admission_status == "passed"
    assert registry.admit(fp8_compute).admission_status == "passed"

    with pytest.raises(vp.AdmissionError):
        vpx.AxisRegistry().register(vpx.AxisDescriptor("bad", ("x",), ()))


def test_standard_axis_registry_accepts_concrete_registered_compile_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.compiler, "list_backends", lambda: ["custom_backend"])
    registry = vpx.standard_axis_registry()
    concrete = vp.Candidate(
        "family",
        "concrete-backend",
        {"compile.backend": "custom_backend"},
    )
    placeholder = vp.Candidate(
        "family",
        "placeholder-backend",
        {"compile.backend": "registered_backend"},
    )
    missing = vp.Candidate(
        "family",
        "missing-backend",
        {"compile.backend": "missing_backend"},
    )

    assert registry.admit(concrete).admission_status == "passed"
    assert registry.admit(placeholder).admission_status == "failed"
    assert registry.admit(missing).admission_status == "failed"


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
                "dtype.model_compute": "fp32",
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
            },
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked",
            {"dtype.model_compute": "bf16"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked-frontend",
            {
                "dtype.model_compute": "fp32",
                "attention.frontend": "transformers_flash_attention_2",
            },
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked-kernel",
            {
                "dtype.model_compute": "fp32",
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
        allowed_dtypes=("fp32",),
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
        search_policy=vp.SearchPolicy(strategy="exhaustive"),
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


AUTOBATCH_FAILURE_SIGNALS = (
    "backend_rejection",
    "oom",
    "reference_failure",
    "runtime_failure",
)


def make_autobatch_domain(
    *,
    min_value: int = 1,
    max_value: int = 4,
    initial_value: int = 1,
    growth: str = "doubling",
    values: tuple[int, ...] = (1, 2, 4),
    settings_by_value: Mapping[int, Mapping[str, object]] | None = None,
    objective: str = "fastest_passing",
    failure_signals: tuple[str, ...] = AUTOBATCH_FAILURE_SIGNALS,
    termination: str = "exhausted_declared_values",
) -> vpx.AutobatchDomain:
    selected_settings = (
        {value: {"batch_size": value} for value in values}
        if settings_by_value is None
        else settings_by_value
    )

    return vpx.AutobatchDomain(
        axis_name="batch_size",
        min_value=min_value,
        max_value=max_value,
        initial_value=initial_value,
        growth=growth,
        values=values,
        settings_by_value=selected_settings,
        value_to_settings_id="tests.batch_size_settings",
        admission_identity={"case": "test"},
        objective=objective,
        failure_signals=failure_signals,
        termination=termination,
        warmup_steps=0,
        measure_steps=1,
        devices=(0,),
        cache_key_payload={"case": "autobatch-domain"},
    )


def test_autobatch_domain_signature_records_finite_search_shape() -> None:
    domain = make_autobatch_domain()
    signature = domain.signature()

    assert signature["min_value"] == 1
    assert signature["max_value"] == 4
    assert signature["initial_value"] == 1
    assert signature["growth"] == "doubling"
    assert signature["objective"] == "fastest_passing"
    assert signature["failure_signals"] == AUTOBATCH_FAILURE_SIGNALS
    assert signature["termination"] == "exhausted_declared_values"
    assert signature["values"] == (1, 2, 4)
    assert signature["settings_by_value"] == {
        "1": {"batch_size": 1},
        "2": {"batch_size": 2},
        "4": {"batch_size": 4},
    }


def test_autobatch_domain_accepts_all_growth_rules() -> None:
    doubling = make_autobatch_domain(growth="doubling", values=(1, 2, 4))
    linear = make_autobatch_domain(
        growth="linear_step",
        values=(1, 3, 5),
        max_value=5,
    )
    declared = make_autobatch_domain(
        growth="declared_sequence",
        values=(1, 3, 8),
        max_value=8,
    )

    assert doubling.signature()["growth"] == "doubling"
    assert linear.signature()["growth"] == "linear_step"
    assert declared.signature()["growth"] == "declared_sequence"


def test_autobatch_domain_rejects_invalid_finite_search_shape() -> None:
    with pytest.raises(RuntimeError, match="strictly increasing"):
        make_autobatch_domain(values=(1, 1, 2))

    with pytest.raises(RuntimeError, match="doubling"):
        make_autobatch_domain(values=(1, 3, 4))

    with pytest.raises(RuntimeError, match="initial_value"):
        make_autobatch_domain(initial_value=2)

    with pytest.raises(RuntimeError, match="settings_by_value"):
        make_autobatch_domain(settings_by_value={1: {"batch_size": 1}})

    with pytest.raises(RuntimeError, match="objective"):
        make_autobatch_domain(objective="fastest_step")

    with pytest.raises(RuntimeError, match="bracketed_failure_frontier"):
        make_autobatch_domain(
            objective="largest_passing",
            termination="exhausted_declared_values",
        )


def test_tune_fast_strategy_delegates_autobatch_domain_to_autobatch_find(
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

    target = dataclasses.replace(
        cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        search_policy=vp.SearchPolicy(strategy="fast"),
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
                    min_value=1,
                    max_value=2,
                    initial_value=1,
                    growth="linear_step",
                    values=(1, 2),
                    settings_by_value={
                        1: {"batch_size": 1},
                        2: {"batch_size": 2},
                    },
                    value_to_settings_id="tests.batch_size_settings",
                    admission_identity={"case": "test"},
                    objective="fastest_passing",
                    failure_signals=(
                        "backend_rejection",
                        "oom",
                        "reference_failure",
                        "runtime_failure",
                    ),
                    termination="exhausted_declared_values",
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


def test_autobatch_domain_filters_reference_failures_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = []
    find_values = []
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        if candidate.settings["batch_size"] == 1:
            message = "reference rejected value"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            probed.append(candidate.settings["batch_size"])

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
        assert goal == autobatch.Goal.largest_safe()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        find_values.append(tuple(values))
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
                    min_value=1,
                    max_value=2,
                    initial_value=1,
                    growth="linear_step",
                    values=(1, 2),
                    settings_by_value={
                        1: {"batch_size": 1},
                        2: {"batch_size": 2},
                    },
                    value_to_settings_id="tests.batch_size_settings",
                    admission_identity={"case": "test"},
                    objective="largest_passing",
                    failure_signals=AUTOBATCH_FAILURE_SIGNALS,
                    termination="bracketed_failure_frontier",
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
        clock=SequenceClock((0.0, 2.0)),
    )

    failed = {
        record.candidate_id: record
        for record in plan.full_size_records
        if record.status == "failed"
    }

    assert find_values == [(2,)]
    assert probed == [2]
    assert plan.selected["family"].candidate_id == "base|batch_size=2"
    assert failed["base|batch_size=1"].error_type == "ReferenceFailed"


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
                    min_value=1,
                    max_value=2,
                    initial_value=1,
                    growth="linear_step",
                    values=(1, 2),
                    settings_by_value={
                        1: {"batch_size": 1},
                        2: {"batch_size": 2},
                    },
                    value_to_settings_id="tests.batch_size_settings",
                    admission_identity={"case": "test"},
                    objective="largest_passing",
                    failure_signals=(
                        "backend_rejection",
                        "oom",
                        "reference_failure",
                        "runtime_failure",
                    ),
                    termination="bracketed_failure_frontier",
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


def test_reference_failed_rows_are_rechecked_on_next_tune(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")
    calls = {"reference": 0, "operation": 0}

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls["reference"] += 1

        if calls["reference"] == 1:
            message = "reference rejected row"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls["operation"] += 1

            return vector

        return operation

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
            {"generator": "reference-rerun"},
        ),
    )

    with pytest.raises(vp.NoPassedCandidateError):
        vp.tune(
            problem,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    failed_reference = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close.json"
    )
    passed_reference = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close-000001.json"
    )
    failed_full_size = read_record(
        tmp_path / "full_size" / "family" / "row" / "result.json"
    )
    passed_full_size = read_record(
        tmp_path / "full_size" / "family" / "row" / "result-000001.json"
    )

    assert calls == {"reference": 2, "operation": 1}
    assert plan.selected_candidate().candidate_id == "row"
    assert failed_reference["status"] == "failed"
    assert failed_reference["error_type"] == "ReferenceFailed"
    assert passed_reference["status"] == "passed"
    assert failed_full_size["status"] == "failed"
    assert failed_full_size["error_type"] == "ReferenceFailed"
    assert passed_full_size["status"] == "passed"


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
        {"dtype": "fp32"},
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


def test_plan_replay_rejects_missing_full_size_agreement() -> None:
    input_signature = _input_signature("full-size-gate-replay")
    candidate = vp.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
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

    with pytest.raises(vp.VPTuneError, match=r"accepted rows|did not pass"):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
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
    assert failed["error_type"] == "ReferenceFailed"
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
        distribution="explicit_score_gradients",
        label_policy="explicit_scores",
        sample_space="terms",
        score_reduction="none",
        denominator="batch_normalization",
    )

    assert fisher.kind == "fisher_vp"
    assert fisher.semantics["distribution"] == "explicit_score_gradients"
    assert fisher.semantics["score_reduction"] == "none"
    assert fisher.batch_inputs == {"reference": (), "operation": ()}
    with pytest.raises(vp.MaterializationError, match="GGNVP"):
        vp.fisher_vp(
            "metric",
            "retain",
            aggregation="mean",
            distribution="categorical",
            label_policy="model_distribution",
            sample_space="classes",
            score_reduction="none",
            denominator="num_examples",
        )

    sampled = vp.sampled_fisher_vp(
        "metric",
        "retain",
        aggregation="mean",
        distribution="explicit_score_gradients",
        label_policy="sampled_labels",
        sample_count=8,
        sample_source="fixed_seed_and_count",
        sampling_bound={"gamma": 0.25},
        score_reduction="none",
        denominator="num_examples",
    )

    assert sampled.kind == "sampled_fisher_vp"
    assert sampled.semantics == {
        "distribution": "explicit_score_gradients",
        "label_policy": "sampled_labels",
        "sample_count": 8,
        "sample_source": "fixed_seed_and_count",
        "sampling_bound": {"gamma": 0.25},
        "score_reduction": "none",
        "denominator": "num_examples",
    }
    assert sampled.batch_inputs == {"reference": (), "operation": ()}
    with pytest.raises(vp.MaterializationError, match="sample_count"):
        vp.sampled_fisher_vp(
            "metric",
            "retain",
            aggregation="mean",
            distribution="explicit_score_gradients",
            label_policy="sampled_labels",
            sample_count=0,
            sample_source="fixed_seed_and_count",
            sampling_bound={"gamma": 0.25},
            score_reduction="none",
            denominator="num_examples",
        )
    with pytest.raises(vp.MaterializationError, match="sample_source"):
        vp.sampled_fisher_vp(
            "metric",
            "retain",
            aggregation="mean",
            distribution="explicit_score_gradients",
            label_policy="sampled_labels",
            sample_count=1,
            sample_source="live_random_samples",
            sampling_bound={"gamma": 0.25},
            score_reduction="none",
            denominator="num_examples",
        )

    empirical = vp.empirical_fisher_vp(
        "metric",
        "retain",
        aggregation="mean",
        example_loss_reduction="per_example",
        denominator="num_examples",
    )

    assert empirical.kind == "empirical_fisher_vp"
    assert empirical.semantics == {
        "example_loss_reduction": "per_example",
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
    dense_representation = {"kind": "dense_matrix"}
    metric = vp.metric(
        "metric",
        "retain",
        aggregation="mean",
        representation=dense_representation,
    )
    inverse_metric = vp.inverse_metric(
        "inverse_metric",
        "retain",
        aggregation="mean",
        representation=dense_representation,
        damping=0.0,
    )

    assert jvp.kind == "jvp"
    assert jvp.batch_inputs == {"reference": (), "operation": ()}
    assert vjp.kind == "vjp"
    assert vjp.batch_inputs == {"reference": ("tangent_vector",), "operation": ()}
    assert metric.kind == "metric"
    assert metric.semantics == {"representation": dense_representation}
    assert metric.batch_inputs == {
        "reference": ("metric_matrix",),
        "operation": ("metric_matrix",),
    }
    assert inverse_metric.kind == "inverse_metric"
    assert inverse_metric.semantics == {
        "damping": 0.0,
        "representation": dense_representation,
    }
    assert inverse_metric.batch_inputs == {
        "reference": ("metric_matrix",),
        "operation": ("metric_matrix",),
    }
    with pytest.raises(vp.MaterializationError, match="damping"):
        vp.inverse_metric(
            "inverse_metric",
            "retain",
            aggregation="mean",
            representation=dense_representation,
            damping=-1.0,
        )
    composition = vp.composition(
        "compose",
        "hvp_after_metric",
        aggregation="none",
        children=("metric", "hvp"),
    )
    assert composition.kind == "composition"
    assert composition.semantics == {"children": ("metric", "hvp")}
    assert vp.Family("compose", composition).dependencies == ("metric", "hvp")

    with pytest.raises(vp.MaterializationError, match="derived from children"):
        vp.Family("compose", composition, dependencies=("metric",))


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
                        {"dtype.model_compute": "fp16"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "a",
                        "a-float32",
                        {"dtype.model_compute": "fp32"},
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
                        {"dtype.model_compute": "fp16"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "b",
                        "b-float32",
                        {"dtype.model_compute": "fp32"},
                        admission_status="passed",
                    ),
                ),
            ),
        ),
        cohort_constraints=(
            vp.CohortConstraint(
                name="dtype",
                settings_keys=("dtype.model_compute",),
                assignments=(
                    {"dtype.model_compute": "fp16"},
                    {"dtype.model_compute": "fp32"},
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
    assert plan.cohort_assignment.values == {"dtype.model_compute": "fp32"}


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
