import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch

import vptune as vp
from vptune.io import read_record
from vptune.measure import CPUMemoryBackend
from vptune.tensor_tree import tree_leaves, tree_map


class OneParameterModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))


class TwoParameterModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))


class ScaleData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "scale"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name, "scale": 2.0}

    @staticmethod
    def probe_batches(family: str) -> Sequence[Mapping[str, object]]:
        return ({"family": family, "scale": 2.0},)


class ParameterVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "parameter_vector"}

    @staticmethod
    def reference_vectors(family: str) -> vp.TensorTree:
        assert family

        return {"w": torch.tensor([3.0], dtype=torch.float64)}

    @staticmethod
    def probe_vectors(family: str) -> Sequence[vp.TensorTree]:
        assert family

        return ({"w": torch.tensor([3.0], dtype=torch.float64)},)


class DenseMetricData:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)

    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "dense_metric"}

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"metric": cls.matrix}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"metric": cls.matrix},)


class TwoParameterVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_parameter_vector"}

    @staticmethod
    def reference_vectors(family: str) -> vp.TensorTree:
        assert family

        return {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    @staticmethod
    def probe_vectors(family: str) -> Sequence[vp.TensorTree]:
        assert family

        return ({"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},)


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1

        return value


def cpu_target() -> vp.Target:
    return vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float64", "float32", "bfloat16", "float16"),
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


def replay_context_for_plan(plan: vp.Plan) -> vp.ReplayContext:
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
    )


def quadratic_scalar(
    params: vp.ParameterTree,
    buffers: vp.BufferTree,
    batch: vp.Batch,
    context: vp.ObjectiveContext,
) -> torch.Tensor:
    assert buffers == {}
    assert context.family

    return batch["scale"] * params["w"].pow(2).sum()


def square_function(
    params: vp.ParameterTree,
    buffers: vp.BufferTree,
    batch: vp.Batch,
    context: vp.ObjectiveContext,
) -> vp.TensorTree:
    assert buffers == {}
    assert batch["scale"]
    assert context.family

    return {"y": params["w"].pow(2)}


def multiply_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    scale = batch["scale"]
    assert isinstance(scale, float)

    return tree_map(lambda tensor: tensor * scale, vector)


def shift_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 1.0, vector)


def wrong_shift_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 2.0, vector)


def identity_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    assert batch["scale"]

    return vector


def torch_func_settings(*, requires_forward_ad: bool) -> dict[str, object]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vmap_randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def score_terms_fisher(family: str, objective_id: str) -> vp.OperatorSpec:
    return vp.fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="explicit_scores",
        expectation="explicit_rows",
        sample_space="terms",
        loss_reduction="none",
        denominator="batch_normalization",
    )


def exact_categorical_fisher(family: str, objective_id: str) -> vp.OperatorSpec:
    return vp.fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        distribution="categorical",
        label_policy="model_distribution",
        expectation="exact",
        sample_space="classes",
        loss_reduction="log_prob",
        denominator="num_examples",
        logits_axis=1,
    )


def add_one_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 1.0, vector)


def subtract_one_component(batch: vp.Batch, vector: vp.TensorTree) -> vp.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor - 1.0, vector)


def passed_child_reference(
    candidate: vp.Candidate,
    batch: vp.Batch,
    vector: vp.TensorTree,
) -> vp.ReferenceResult:
    assert candidate.family == "child"
    assert batch["scale"]
    assert vector

    return vp.ReferenceResult("child_anchor", {}, {"child_value": 0.0})


def failed_child_reference(
    candidate: vp.Candidate,
    batch: vp.Batch,
    vector: vp.TensorTree,
) -> vp.ReferenceResult:
    assert candidate.family == "child"
    assert batch["scale"]
    assert vector

    message = "child anchor failed"
    raise vp.ReferenceFailedError(message)


def test_standard_operation_factory_runs_core_derivative_products() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {}
    batch = {"family": "family", "scale": 5.0}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    cotangent = {"y": torch.tensor([4.0], dtype=torch.float64)}

    gradient_factory = vp.standard_operation_factory(
        vp.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": quadratic_scalar},
    )
    jvp_factory = vp.standard_operation_factory(
        vp.jvp("jvp", "function", aggregation="none"),
        params=params,
        buffers=buffers,
        function_objectives={"function": square_function},
    )
    vjp_factory = vp.standard_operation_factory(
        vp.vjp("vjp", "function", aggregation="none"),
        params=params,
        buffers=buffers,
        function_objectives={"function": square_function},
    )
    hvp_factory = vp.standard_operation_factory(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": quadratic_scalar},
    )
    gradient = gradient_factory(
        vp.Candidate(
            "gradient",
            "row",
            {"operator_path": "autograd_grad"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    jvp = jvp_factory(
        vp.Candidate(
            "jvp",
            "row",
            {"operator_path": "torch_func_jvp"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    vjp = vjp_factory(
        vp.Candidate(
            "vjp",
            "row",
            {"operator_path": "torch_func_vjp"},
            admission_status="passed",
        ),
        batch,
        cotangent,
    )()
    hvp = hvp_factory(
        vp.Candidate(
            "hvp",
            "row",
            {"operator_path": "jvp_grad"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    hvp_from_vhp = hvp_factory(
        vp.Candidate(
            "hvp",
            "row",
            {"operator_path": "vhp"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    assert torch.allclose(
        tree_leaves(gradient)[0],
        torch.tensor([20.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(jvp)[0],
        torch.tensor([12.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(vjp)[0],
        torch.tensor([16.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(hvp)[0],
        torch.tensor([30.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(hvp_from_vhp)[0],
        torch.tensor([30.0], dtype=torch.float64),
    )


def test_standard_reference_check_requires_thresholds() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    with pytest.raises(vp.MaterializationError):
        vp.standard_reference_check(
            vp.hvp("hvp", "loss", aggregation="sum"),
            params=params,
            buffers={},
            thresholds={},
            scalar_objectives={"loss": quadratic_scalar},
        )


def test_standard_reference_check_preserves_candidate_context_identity() -> None:
    calls = []
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    def scalar(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"]
        calls.append(context.candidate_id)

        return params["w"].pow(2).sum()

    check = vp.standard_reference_check(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
        },
        scalar_objectives={"loss": scalar},
    )
    candidate = vp.Candidate(
        "hvp",
        "candidate-row",
        {"operator_path": "jvp_grad"},
        admission_status="passed",
    )

    result = check(
        candidate,
        {"scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert calls
    assert all(call == "candidate-row" for call in calls)
    assert "directional_abs_diff" in result.measurements
    assert "directional_rel_diff" in result.measurements


def test_hvp_reference_check_records_symmetry_error() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
            "symmetry_max_abs_diff": 1e-9,
        },
        scalar_objectives={"loss": quadratic_scalar},
    )
    result = check(
        vp.Candidate(
            "hvp",
            "row",
            {"operator_path": "jvp_grad"},
            admission_status="passed",
        ),
        {
            "scale": 1.0,
            "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["symmetry_max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "hvp",
                "row",
                {"operator_path": "jvp_grad"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_gradient_reference_check_records_directional_agreement() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
        },
        scalar_objectives={"loss": quadratic_scalar},
    )
    result = check(
        vp.Candidate(
            "gradient",
            "row",
            {"operator_path": "autograd_grad"},
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["directional_abs_diff"] < 1e-3


def test_jvp_reference_check_records_finite_difference_agreement() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.jvp("jvp", "function", aggregation="none"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
        },
        function_objectives={"function": square_function},
    )
    result = check(
        vp.Candidate(
            "jvp",
            "row",
            {"operator_path": "torch_func_jvp"},
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["directional_abs_diff"] < 1e-3


def test_vjp_reference_check_records_dot_identity() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.vjp("vjp", "function", aggregation="none"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "inner_abs_diff": 1e-9,
        },
        function_objectives={"function": square_function},
    )
    result = check(
        vp.Candidate(
            "vjp",
            "row",
            {"operator_path": "torch_func_vjp"},
            admission_status="passed",
        ),
        {
            "scale": 1.0,
            "tangent_vector": {"w": torch.tensor([3.0], dtype=torch.float64)},
        },
        {"y": torch.tensor([4.0], dtype=torch.float64)},
    )

    assert result.measurements["inner_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "vjp",
                "row",
                {"operator_path": "torch_func_vjp"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"y": torch.tensor([4.0], dtype=torch.float64)},
        )


def test_vhp_reference_check_requires_symmetry_and_directional_checks() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check_without_symmetry = vp.standard_reference_check(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
        },
        scalar_objectives={"loss": quadratic_scalar},
    )
    check_with_symmetry = vp.standard_reference_check(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
            "symmetry_max_abs_diff": 1e-9,
        },
        scalar_objectives={"loss": quadratic_scalar},
    )
    candidate = vp.Candidate(
        "hvp",
        "vhp",
        {"operator_path": "vhp"},
        admission_status="passed",
    )

    with pytest.raises(vp.ReferenceFailedError):
        check_without_symmetry(
            candidate,
            {
                "scale": 1.0,
                "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError):
        check_with_symmetry(
            candidate,
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_standard_reference_check_honors_strict_thresholds() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    def scalar(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"]
        multiplier = (
            1.000005 if context.settings["operator_path"] == "jvp_grad" else 1.0
        )

        return multiplier * params["w"].pow(2).sum()

    check = vp.standard_reference_check(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar},
    )
    candidate = vp.Candidate(
        "hvp",
        "candidate-row",
        {"operator_path": "jvp_grad"},
        admission_status="passed",
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            candidate,
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_reference_check_low_precision_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "metric",
                "float32",
                {"operator_path": "dense_metric", "model_dtype": "float32"},
                admission_status="passed",
            ),
            {"metric": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.00000006], dtype=torch.float64)},
        )


def test_metric_reference_check_rejects_nonsymmetric_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "metric",
                "row",
                {"operator_path": "dense_metric"},
                admission_status="passed",
            ),
            {"metric": torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)},
            {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        )


def test_metric_reference_check_rejects_indefinite_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "metric",
                "row",
                {"operator_path": "dense_metric"},
                admission_status="passed",
            ),
            {"metric": torch.diag(torch.tensor([1.0, -0.1], dtype=torch.float64))},
            {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        )


def test_dense_standard_paths_reject_nonfinite_inputs() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return params["w"]

    ggn_factory = vp.standard_operation_factory(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    fisher_factory = vp.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    metric_factory = vp.standard_operation_factory(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="nonfinite"):
        ggn_factory(
            vp.Candidate(
                "ggn",
                "row",
                {"operator_path": "dense_ggn"},
                admission_status="passed",
            ),
            {"loss_hessian": torch.tensor([[torch.nan]], dtype=torch.float64)},
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="nonfinite"):
        fisher_factory(
            vp.Candidate(
                "fisher",
                "row",
                {"operator_path": "dense_score_outer"},
                admission_status="passed",
            ),
            {
                "score_gradients": torch.tensor([[torch.inf]], dtype=torch.float64),
                "normalization": 1.0,
            },
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="nonfinite"):
        metric_factory(
            vp.Candidate(
                "metric",
                "row",
                {"operator_path": "dense_metric"},
                admission_status="passed",
            ),
            {"metric": torch.tensor([[torch.nan]], dtype=torch.float64)},
            vector,
        )()


def test_inverse_metric_reference_check_records_inverse_residual() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.inverse_metric("inverse", "dense", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
        },
    )
    result = check(
        vp.Candidate(
            "inverse",
            "row",
            {"operator_path": "dense_inverse_metric"},
            admission_status="passed",
        ),
        {"metric": torch.eye(2, dtype=torch.float64)},
        {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
    )

    assert result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert result.measurements["psd_violation"] == pytest.approx(0.0)


def test_inverse_metric_reference_check_rejects_indefinite_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vp.standard_reference_check(
        vp.inverse_metric("inverse", "dense", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "inverse",
                "row",
                {"operator_path": "dense_inverse_metric"},
                admission_status="passed",
            ),
            {"metric": torch.diag(torch.tensor([1.0, -0.1], dtype=torch.float64))},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_rejects_nonsymmetric_loss_hessian() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 2.0 * params["w"][0]))

    check = vp.standard_reference_check(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "ggn",
                "row",
                {"operator_path": "dense_ggn"},
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.tensor(
                    [[1.0, 2.0], [0.0, 1.0]],
                    dtype=torch.float64,
                )
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_rejects_indefinite_loss_hessian() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 2.0 * params["w"][0]))

    check = vp.standard_reference_check(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "ggn",
                "row",
                {"operator_path": "dense_ggn"},
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.diag(
                    torch.tensor(
                        [1.0, -0.1],
                        dtype=torch.float64,
                    )
                )
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_uses_jvp_hessian_vjp_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        multiplier = 2.0 if context.settings["operator_path"] == "dense_ggn" else 1.0

        return multiplier * torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vp.standard_reference_check(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "ggn",
                "row",
                {"operator_path": "dense_ggn"},
                admission_status="passed",
            ),
            {"loss_hessian": torch.eye(2, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_cross_checks_jvp_path_with_dense_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        multiplier = 2.0 if context.settings["operator_path"] == "dense_ggn" else 1.0

        return multiplier * torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vp.standard_reference_check(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "ggn",
                "row",
                {"operator_path": "jvp_hessian_vjp"},
                admission_status="passed",
            ),
            {"loss_hessian": torch.eye(2, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_records_dense_anchor_errors() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vp.standard_reference_check(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
        function_objectives={"model_output": function},
    )
    result = check(
        vp.Candidate(
            "ggn",
            "row",
            {"operator_path": "jvp_hessian_vjp"},
            admission_status="passed",
        ),
        {"loss_hessian": torch.eye(2, dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert result.measurements["dense_anchor_errors"] == {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }


def test_fisher_references_use_declared_per_example_objectives() -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}

    def per_example_scores(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["normalization"] == pytest.approx(3.0)
        assert context.family in {"fisher", "empirical"}

        return torch.stack((
            params["w"][0],
            2.0 * params["w"][0] - params["w"][1],
            0.5 * params["w"][0] + 3.0 * params["w"][1],
        ))

    score_gradients = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    wrong_gradients = torch.zeros_like(score_gradients)
    thresholds = {"max_abs_diff": 1e-12, "max_rel_diff": 1e-12}
    fisher_check = vp.standard_reference_check(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        thresholds=thresholds,
        function_objectives={"scores": per_example_scores},
    )
    empirical_check = vp.standard_reference_check(
        vp.empirical_fisher_vp(
            "empirical",
            "scores",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers={},
        thresholds=thresholds,
        function_objectives={"scores": per_example_scores},
    )
    fisher_result = fisher_check(
        vp.Candidate(
            "fisher",
            "row",
            {"operator_path": "dense_score_outer"},
            admission_status="passed",
        ),
        {"score_gradients": score_gradients, "normalization": 3.0},
        vector,
    )
    empirical_result = empirical_check(
        vp.Candidate(
            "empirical",
            "row",
            {"operator_path": "dense_empirical_fisher"},
            admission_status="passed",
        ),
        {"per_example_gradients": score_gradients, "normalization": 3.0},
        vector,
    )

    assert fisher_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert empirical_result.measurements["max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError):
        fisher_check(
            vp.Candidate(
                "fisher",
                "row",
                {"operator_path": "dense_score_outer"},
                admission_status="passed",
            ),
            {"score_gradients": wrong_gradients, "normalization": 3.0},
            vector,
        )

    with pytest.raises(vp.ReferenceFailedError):
        empirical_check(
            vp.Candidate(
                "empirical",
                "row",
                {"operator_path": "dense_empirical_fisher"},
                admission_status="passed",
            ),
            {"per_example_gradients": wrong_gradients, "normalization": 3.0},
            vector,
        )


def test_exact_categorical_fisher_differs_from_empirical_fisher() -> None:
    params = {"w": torch.tensor([0.7], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.3], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([2.0], dtype=torch.float64),
        "labels": torch.tensor([0]),
    }

    def logits(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"
        x_value = batch["x"]
        assert isinstance(x_value, torch.Tensor)

        return torch.stack((params["w"][0] * x_value, torch.zeros_like(x_value)), dim=1)

    def observed_losses(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "empirical"
        x_value = batch["x"]
        labels = batch["labels"]
        assert isinstance(x_value, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        log_probs = torch.log_softmax(
            torch.stack(
                (params["w"][0] * x_value, torch.zeros_like(x_value)),
                dim=1,
            ),
            dim=1,
        )

        return -log_probs[torch.arange(labels.numel()), labels]

    fisher_factory = vp.standard_operation_factory(
        exact_categorical_fisher("fisher", "logits"),
        params=params,
        buffers={},
        function_objectives={"logits": logits},
    )
    empirical_factory = vp.standard_operation_factory(
        vp.empirical_fisher_vp(
            "empirical",
            "losses",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers={},
        function_objectives={"losses": observed_losses},
    )
    fisher_result = fisher_factory(
        vp.Candidate(
            "fisher",
            "exact",
            {"operator_path": "categorical_exact"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    empirical_result = empirical_factory(
        vp.Candidate(
            "empirical",
            "loop",
            {"operator_path": "per_example_gradient_loop"},
            admission_status="passed",
        ),
        {**batch, "normalization": 1.0},
        vector,
    )()
    x_scalar = batch["x"][0]
    p0 = torch.softmax(
        torch.stack((
            params["w"][0] * x_scalar,
            torch.tensor(0.0, dtype=torch.float64),
        )),
        dim=0,
    )[0]
    expected_fisher = x_scalar.pow(2) * p0 * (1.0 - p0) * vector["w"]
    expected_empirical = x_scalar.pow(2) * (1.0 - p0).pow(2) * vector["w"]

    assert torch.allclose(tree_leaves(fisher_result)[0], expected_fisher)
    assert torch.allclose(tree_leaves(empirical_result)[0], expected_empirical)
    assert not torch.allclose(
        tree_leaves(fisher_result)[0], tree_leaves(empirical_result)[0]
    )


def test_empirical_fisher_vmap_path_matches_loop_path() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
        "y": torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64),
        "normalization": 3.0,
    }

    def per_example_losses(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "empirical"
        x_value = batch["x"]
        y_value = batch["y"]
        assert isinstance(x_value, torch.Tensor)
        assert isinstance(y_value, torch.Tensor)

        return (params["w"][0] * x_value - y_value).square()

    factory = vp.standard_operation_factory(
        vp.empirical_fisher_vp(
            "empirical",
            "losses",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers={},
        function_objectives={"losses": per_example_losses},
    )
    loop_result = factory(
        vp.Candidate(
            "empirical",
            "loop",
            {"operator_path": "per_example_gradient_loop"},
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    vmap_result = factory(
        vp.Candidate(
            "empirical",
            "vmap",
            {
                "operator_path": "per_example_gradient_vmap",
                "vmap_chunk_size": 1,
                "vmap_batch_in_dims": {
                    "x": 0,
                    "y": 0,
                    "normalization": None,
                },
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    check = vp.standard_reference_check(
        vp.empirical_fisher_vp(
            "empirical",
            "losses",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers={},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
        function_objectives={"losses": per_example_losses},
    )
    reference_result = check(
        vp.Candidate(
            "empirical",
            "vmap-reference",
            {
                "operator_path": "per_example_gradient_vmap",
                "vmap_chunk_size": 1,
                "vmap_batch_in_dims": {
                    "x": 0,
                    "y": 0,
                    "normalization": None,
                },
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )

    assert torch.allclose(tree_leaves(vmap_result)[0], tree_leaves(loop_result)[0])
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.MaterializationError, match="vmap_chunk_size"):
        factory(
            vp.Candidate(
                "empirical",
                "bad-loop",
                {
                    "operator_path": "per_example_gradient_loop",
                    "vmap_chunk_size": 1,
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="vmap_chunk_size"):
        factory(
            vp.Candidate(
                "empirical",
                "missing-chunk",
                {
                    "operator_path": "per_example_gradient_vmap",
                    "vmap_batch_in_dims": {
                        "x": 0,
                        "y": 0,
                        "normalization": None,
                    },
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="vmap_batch_in_dims"):
        factory(
            vp.Candidate(
                "empirical",
                "missing-in-dims",
                {
                    "operator_path": "per_example_gradient_vmap",
                    "vmap_chunk_size": 1,
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()


def test_empirical_fisher_vmap_path_rejects_invalid_batch_shape() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}

    def per_example_losses(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert params
        assert buffers == {}
        assert context.family == "empirical"
        x_value = batch["x"]
        assert isinstance(x_value, torch.Tensor)

        return x_value * 0.0

    factory = vp.standard_operation_factory(
        vp.empirical_fisher_vp(
            "empirical",
            "losses",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers={},
        function_objectives={"losses": per_example_losses},
    )

    with pytest.raises(vp.MaterializationError, match="leading dimensions differ"):
        factory(
            vp.Candidate(
                "empirical",
                "vmap",
                {
                    "operator_path": "per_example_gradient_vmap",
                    "vmap_chunk_size": 1,
                    "vmap_batch_in_dims": {
                        "x": 0,
                        "y": 0,
                        "normalization": None,
                    },
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {
                "x": torch.tensor([1.0, 2.0], dtype=torch.float64),
                "y": torch.tensor([1.0], dtype=torch.float64),
                "normalization": 2.0,
            },
            vector,
        )()


def test_exact_categorical_fisher_anchor_checks_precomputed_score_rows() -> None:
    params = {"w": torch.tensor([0.7], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.3], dtype=torch.float64)}

    def logits(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"
        x_value = batch["x"]
        assert isinstance(x_value, torch.Tensor)

        return torch.stack((params["w"][0] * x_value, torch.zeros_like(x_value)), dim=1)

    check = vp.standard_reference_check(
        exact_categorical_fisher("fisher", "logits"),
        params=params,
        buffers={},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
        function_objectives={"logits": logits},
    )
    x_scalar = torch.tensor(2.0, dtype=torch.float64)
    p0 = torch.softmax(
        torch.stack((
            params["w"][0] * x_scalar,
            torch.tensor(0.0, dtype=torch.float64),
        )),
        dim=0,
    )[0]
    score_gradients = torch.stack((
        torch.sqrt(p0) * x_scalar * (1.0 - p0),
        torch.sqrt(1.0 - p0) * (-x_scalar * p0),
    )).reshape(2, 1)
    result = check(
        vp.Candidate(
            "fisher",
            "dense",
            {"operator_path": "dense_score_outer"},
            admission_status="passed",
        ),
        {
            "x": torch.tensor([2.0], dtype=torch.float64),
            "score_gradients": score_gradients,
            "num_examples": 1,
        },
        vector,
    )

    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "fisher",
                "dense",
                {"operator_path": "dense_score_outer"},
                admission_status="passed",
            ),
            {
                "x": torch.tensor([2.0], dtype=torch.float64),
                "score_gradients": torch.zeros_like(score_gradients),
                "num_examples": 1,
            },
            vector,
        )


def test_standard_operation_factory_runs_dense_metric_and_fisher_families() -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    buffers = {}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    score_gradients = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] + 2.0 * params["w"][1],
            -params["w"][0] + params["w"][1],
        ))

    ggn_factory = vp.standard_operation_factory(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers=buffers,
        function_objectives={"model_output": function},
    )
    fisher_factory = vp.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers=buffers,
    )
    empirical_factory = vp.standard_operation_factory(
        vp.empirical_fisher_vp(
            "empirical",
            "scores",
            aggregation="mean_per_example",
        ),
        params=params,
        buffers=buffers,
    )
    metric_factory = vp.standard_operation_factory(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers=buffers,
    )
    inverse_factory = vp.standard_operation_factory(
        vp.inverse_metric("inverse", "dense", aggregation="sum"),
        params=params,
        buffers=buffers,
    )
    ggn_result = ggn_factory(
        vp.Candidate(
            "ggn",
            "row",
            {"operator_path": "dense_ggn"},
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    fisher_result = fisher_factory(
        vp.Candidate(
            "fisher",
            "row",
            {"operator_path": "dense_score_outer"},
            admission_status="passed",
        ),
        {"score_gradients": score_gradients, "normalization": 3.0},
        vector,
    )()
    empirical_result = empirical_factory(
        vp.Candidate(
            "empirical",
            "row",
            {"operator_path": "dense_empirical_fisher"},
            admission_status="passed",
        ),
        {"per_example_gradients": score_gradients, "normalization": 3.0},
        vector,
    )()
    metric_result = metric_factory(
        vp.Candidate(
            "metric",
            "row",
            {"operator_path": "dense_metric"},
            admission_status="passed",
        ),
        {"metric": matrix},
        vector,
    )()
    inverse_result = inverse_factory(
        vp.Candidate(
            "inverse",
            "row",
            {"operator_path": "dense_inverse_metric"},
            admission_status="passed",
        ),
        {"metric": matrix},
        vector,
    )()
    flat_vector = vector["w"]
    jacobian = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    expected_ggn = jacobian.T @ (loss_hessian @ (jacobian @ flat_vector))
    expected_fisher = score_gradients.T @ (score_gradients @ flat_vector) / 3.0

    assert torch.allclose(tree_leaves(ggn_result)[0], expected_ggn)
    assert torch.allclose(tree_leaves(fisher_result)[0], expected_fisher)
    assert torch.allclose(tree_leaves(empirical_result)[0], expected_fisher)
    assert torch.allclose(tree_leaves(metric_result)[0], matrix @ flat_vector)
    assert torch.allclose(
        tree_leaves(inverse_result)[0],
        torch.linalg.solve(matrix, flat_vector),
    )


def test_standard_metric_materializer_returns_metric_object(tmp_path: Path) -> None:
    model = TwoParameterModule()
    params = {"w": model.w.detach().clone()}
    operator = vp.metric("metric", "dense", aggregation="sum")
    candidates = (
        vp.Candidate(
            "metric",
            "dense",
            {"operator_path": "dense_metric"},
            changed_axes=("operator_path",),
        ),
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=DenseMetricData(),
        operator=operator,
        vectors=TwoParameterVectorProvider(),
        target=cpu_target(),
        runtime=vp.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=candidates,
            thresholds={
                "max_abs_diff": 1e-12,
                "max_rel_diff": 1e-12,
                "symmetry_max_abs_diff": 1e-12,
                "psd_violation": 1e-12,
            },
            objective_signature={"dense": "metric-v1"},
            axis_registry=vp.standard_axis_registry(),
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, family="metric")
    batch = {"metric": DenseMetricData.matrix}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([-1.0, 0.5], dtype=torch.float64)}

    assert isinstance(selected, vp.StandardMetricOperator)
    assert torch.allclose(
        tree_leaves(selected(batch, vector))[0],
        DenseMetricData.matrix @ vector["w"],
    )
    assert torch.allclose(
        tree_leaves(selected.multiply(batch, vector))[0],
        DenseMetricData.matrix @ vector["w"],
    )
    assert torch.allclose(
        tree_leaves(selected.inverse_multiply(batch, vector))[0],
        torch.linalg.solve(DenseMetricData.matrix, vector["w"]),
    )
    assert torch.allclose(
        selected.inner(batch, vector, right),
        vector["w"] @ (DenseMetricData.matrix @ right["w"]),
    )


def flatten_tree(tree: vp.TensorTree) -> torch.Tensor:
    return torch.cat(tuple(value.reshape(-1) for value in tree_leaves(tree)))


def test_kfac_metric_operator_matches_dense_reference_for_single_block() -> None:
    left = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    right = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    dense = torch.kron(left, right)
    batch = {"a": left, "b": right}
    vector = {"w": torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)}
    other = {"w": torch.tensor([[0.25, -0.75], [1.5, 2.0]], dtype=torch.float64)}
    metric = vp.KFACMetricOperator((vp.KFACMetricBlock("w", "a", "b"),))

    product = flatten_tree(metric.multiply(batch, vector))
    inverse_product = flatten_tree(
        metric.inverse_multiply(batch, vector),
    )
    expected_product = dense @ vector["w"].reshape(-1)
    expected_inverse = torch.linalg.solve(dense, vector["w"].reshape(-1))
    expected_inner = other["w"].reshape(-1) @ (dense @ vector["w"].reshape(-1))

    assert torch.allclose(product, expected_product)
    assert torch.allclose(inverse_product, expected_inverse)
    assert torch.allclose(metric.inner(batch, other, vector), expected_inner)


def test_kfac_metric_operator_matches_block_diagonal_dense_reference() -> None:
    left_w = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    right_w = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    left_b = torch.tensor([[2.0]], dtype=torch.float64)
    right_b = torch.tensor([[5.0, 0.25], [0.25, 2.0]], dtype=torch.float64)
    dense_w = torch.kron(left_w, right_w)
    dense_b = torch.kron(left_b, right_b)
    dense = torch.block_diag(dense_w, dense_b)
    batch = {
        "w_left": left_w,
        "w_right": right_w,
        "b_left": left_b,
        "b_right": right_b,
    }
    vector = {
        "w": torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64),
        "b": torch.tensor([[0.25, -0.75]], dtype=torch.float64),
    }
    other = {
        "w": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64),
        "b": torch.tensor([[0.5, 0.6]], dtype=torch.float64),
    }
    metric = vp.KFACMetricOperator((
        vp.KFACMetricBlock("w", "w_left", "w_right"),
        vp.KFACMetricBlock("b", "b_left", "b_right"),
    ))
    flat_vector = torch.cat((vector["w"].reshape(-1), vector["b"].reshape(-1)))
    flat_other = torch.cat((other["w"].reshape(-1), other["b"].reshape(-1)))
    product = flatten_tree(metric.multiply(batch, vector))
    inverse_product = flatten_tree(metric.inverse_multiply(batch, vector))

    assert torch.allclose(product, dense @ flat_vector)
    assert torch.allclose(inverse_product, torch.linalg.solve(dense, flat_vector))
    assert torch.allclose(metric.inner(batch, other, vector), flat_other @ product)


def test_fisher_style_runtime_rejects_invalid_normalization() -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}
    score_gradients = torch.eye(2, dtype=torch.float64)
    mean_factory = vp.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    sum_factory = vp.standard_operation_factory(
        vp.empirical_fisher_vp("empirical", "scores", aggregation="sum"),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError):
        mean_factory(
            vp.Candidate(
                "fisher",
                "row",
                {"operator_path": "dense_score_outer"},
                admission_status="passed",
            ),
            {"score_gradients": score_gradients, "normalization": 0.0},
            vector,
        )()

    with pytest.raises(vp.MaterializationError):
        sum_factory(
            vp.Candidate(
                "empirical",
                "row",
                {"operator_path": "dense_empirical_fisher"},
                admission_status="passed",
            ),
            {"per_example_gradients": score_gradients, "normalization": 2.0},
            vector,
        )()


def test_dense_metric_and_fisher_support_parameter_tree_order() -> None:
    params = {
        "a": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "b": torch.tensor([[0.0], [0.0]], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    }
    matrix = torch.arange(16.0, dtype=torch.float64).reshape(4, 4)
    score_gradients = torch.eye(4, dtype=torch.float64)
    metric_factory = vp.standard_operation_factory(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers={},
    )
    fisher_factory = vp.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    metric_result = metric_factory(
        vp.Candidate(
            "metric",
            "row",
            {"operator_path": "dense_metric"},
            admission_status="passed",
        ),
        {"metric": matrix},
        vector,
    )()
    fisher_result = fisher_factory(
        vp.Candidate(
            "fisher",
            "row",
            {"operator_path": "dense_score_outer"},
            admission_status="passed",
        ),
        {"score_gradients": score_gradients, "normalization": 1.0},
        vector,
    )()
    flat_vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    metric_flat = matrix @ flat_vector
    metric_leaves = tree_leaves(metric_result)
    fisher_leaves = tree_leaves(fisher_result)
    vector_leaves = tree_leaves(vector)

    assert torch.allclose(metric_leaves[0], metric_flat[:2])
    assert torch.allclose(metric_leaves[1], metric_flat[2:].reshape(2, 1))
    assert torch.allclose(fisher_leaves[0], vector_leaves[0])
    assert torch.allclose(fisher_leaves[1], vector_leaves[1])


def test_dense_ggnvp_supports_parameter_tree_order() -> None:
    params = {
        "a": torch.tensor([0.5, -0.25], dtype=torch.float64),
        "b": torch.tensor([[0.1], [0.2]], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    }
    loss_hessian = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))

    def function(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["a"][0] + params["b"][0, 0],
            params["a"][1] - params["b"][1, 0],
        ))

    factory = vp.standard_operation_factory(
        vp.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vp.Candidate(
            "ggn",
            "row",
            {"operator_path": "dense_ggn"},
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    flat_vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    jacobian = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ flat_vector))
    leaves = tree_leaves(result)

    assert torch.allclose(leaves[0], expected[:2])
    assert torch.allclose(leaves[1], expected[2:].reshape(2, 1))


def test_hvp_vhp_path_supports_parameter_tree_order() -> None:
    params = {
        "a": torch.tensor([0.5, -0.25], dtype=torch.float64),
        "b": torch.tensor([[0.1], [0.2]], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    }

    def scalar(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert isinstance(batch["scale"], float)
        assert context.family == "hvp"

        return batch["scale"] * (params["a"].pow(2).sum() + params["b"].pow(2).sum())

    factory = vp.standard_operation_factory(
        vp.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vp.Candidate(
            "hvp",
            "row",
            {"operator_path": "vhp"},
            admission_status="passed",
        ),
        {"scale": 5.0},
        vector,
    )()
    leaves = tree_leaves(result)

    assert torch.allclose(leaves[0], 10.0 * vector["a"])
    assert torch.allclose(leaves[1], 10.0 * vector["b"])


def test_standard_runtime_executes_dtype_and_backend_axes() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    matrix = torch.eye(1, dtype=torch.float64)
    observed = {}
    metric_factory = vp.standard_operation_factory(
        vp.metric("metric", "dense", aggregation="sum"),
        params=params,
        buffers=buffers,
    )

    def scalar(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert context.family == "gradient"
        observed["param_dtype"] = params["w"].dtype
        observed["buffer_dtype"] = buffers["b"].dtype
        observed["floating_dtype"] = batch["floating"].dtype
        observed["input_ids_dtype"] = batch["input_ids"].dtype
        observed["attention_mask_dtype"] = batch["attention_mask"].dtype
        observed["matmul_precision"] = torch.get_float32_matmul_precision()
        observed["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        observed["allow_bf16"] = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )

        return batch["floating"].sum() * params["w"].pow(2).sum() + buffers["b"].sum()

    gradient_factory = vp.standard_operation_factory(
        vp.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )
    result = metric_factory(
        vp.Candidate(
            "metric",
            "row",
            {"operator_path": "dense_metric", "model_dtype": "float32"},
            admission_status="passed",
        ),
        {"metric": matrix},
        vector,
    )()

    assert tree_leaves(result)[0].dtype == torch.float32

    previous_precision = torch.get_float32_matmul_precision()
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_allow_bf16 = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )

    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        gradient_factory(
            vp.Candidate(
                "gradient",
                "row",
                {
                    "operator_path": "autograd_grad",
                    "model_dtype": "bfloat16",
                    "compute_dtype": "float32",
                    "matmul_precision": "high",
                    "allow_tf32": True,
                    "allow_bf16_reduced_precision_reduction": True,
                },
                admission_status="passed",
            ),
            {
                "floating": torch.tensor([2.0], dtype=torch.float64),
                "input_ids": torch.tensor([1, 2], dtype=torch.long),
                "attention_mask": torch.tensor([True, False]),
            },
            vector,
        )()
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_allow_bf16
        )

    assert observed == {
        "param_dtype": torch.float32,
        "buffer_dtype": torch.float32,
        "floating_dtype": torch.float32,
        "input_ids_dtype": torch.long,
        "attention_mask_dtype": torch.bool,
        "matmul_precision": "high",
        "allow_tf32": True,
        "allow_bf16": True,
    }
    assert torch.get_float32_matmul_precision() == previous_precision
    assert torch.backends.cuda.matmul.allow_tf32 is previous_allow_tf32
    assert (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        is previous_allow_bf16
    )
    functional_result = gradient_factory(
        vp.Candidate(
            "gradient",
            "row",
            {
                "operator_path": "autograd_grad",
                "parameter_keys": ("w",),
                "buffer_keys": ("b",),
                "tie_weights": True,
                "strict": False,
                "parametrization_policy": "active",
                "mutates_state": False,
                "mutated_parameter_keys": (),
                "mutated_buffer_keys": (),
                "module_mode": "eval",
            },
            admission_status="passed",
        ),
        {
            "floating": torch.tensor([1.0], dtype=torch.float64),
            "input_ids": torch.tensor([1], dtype=torch.long),
            "attention_mask": torch.tensor([True]),
        },
        vector,
    )()

    assert torch.equal(tree_leaves(functional_result)[0], torch.tensor([4.0]))

    with pytest.raises(vp.MaterializationError):
        gradient_factory(
            vp.Candidate(
                "gradient",
                "row",
                {"operator_path": "autograd_grad", "batch_size": 2},
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )()


def test_composition_runtime_config_runs_and_materializes_selected_operator(
    tmp_path: Path,
) -> None:
    model = OneParameterModule()
    operator = vp.composition("compose", "scale_then_shift", aggregation="none")
    candidates = (
        vp.Candidate(
            "compose",
            "row",
            {"operator_path": "sequential_composition"},
            admission_status="passed",
        ),
    )
    runtime = vp.composition_runtime_config(
        operator,
        order=("multiply", "shift"),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        anchor_components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        candidates=candidates,
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={
            "multiply": "test.multiply_component",
            "shift": "test.shift_component",
        },
        anchor_component_signature={
            "multiply": "test.multiply_component",
            "shift": "test.shift_component",
        },
        axis_registry=None,
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, family="compose")
    result = selected(
        {"family": "compose", "scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert plan.selected["compose"].candidate_id == "row"
    assert plan.runtime_identities["compose"]["components"] == {
        "candidate": {
            "multiply": "test.multiply_component",
            "shift": "test.shift_component",
        },
        "anchor": {
            "multiply": "test.multiply_component",
            "shift": "test.shift_component",
        },
    }
    assert plan.check_records[0].measurements["component_errors"] == {
        "multiply": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        "shift": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    }
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([7.0], dtype=torch.float64),
    )


def test_composition_reference_check_uses_anchor_components() -> None:
    check = vp.composition_reference_check(
        vp.composition("compose", "scale_then_shift", aggregation="none"),
        order=("multiply", "shift"),
        components={
            "multiply": multiply_component,
            "shift": wrong_shift_component,
        },
        anchor_components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "compose",
                "row",
                {"operator_path": "sequential_composition"},
                admission_status="passed",
            ),
            {"scale": 2.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_reference_check_fails_intermediate_component_mismatch() -> None:
    check = vp.composition_reference_check(
        vp.composition("compose", "hidden_component_error", aggregation="none"),
        order=("first", "second"),
        components={
            "first": add_one_component,
            "second": subtract_one_component,
        },
        anchor_components={
            "first": identity_component,
            "second": identity_component,
        },
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "compose",
                "row",
                {"operator_path": "sequential_composition"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_reference_check_runs_child_anchor() -> None:
    child = vp.CompositionChild(
        name="first",
        candidate=vp.Candidate(
            "child",
            "bad",
            {"operator_path": "child"},
            admission_status="passed",
        ),
        component=add_one_component,
        anchor_component=add_one_component,
        reference_check=failed_child_reference,
        input_signature={"child": "bad"},
    )
    check = vp.composition_reference_check(
        vp.composition("compose", "child_anchor", aggregation="none"),
        order=("first",),
        components={"first": child.component},
        anchor_components={"first": child.anchor_component},
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        children=(child,),
    )

    with pytest.raises(vp.ReferenceFailedError, match="child anchor failed"):
        check(
            vp.Candidate(
                "compose",
                "row",
                {"operator_path": "sequential_composition"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_tune_writes_child_reference_rows(tmp_path: Path) -> None:
    model = OneParameterModule()
    operator = vp.composition("compose", "child_anchor", aggregation="none")
    child = vp.CompositionChild(
        name="identity",
        candidate=vp.Candidate(
            "child",
            "identity",
            {"operator_path": "child"},
            admission_status="passed",
        ),
        component=identity_component,
        anchor_component=identity_component,
        reference_check=passed_child_reference,
        input_signature={"child": "identity"},
    )
    candidates = (
        vp.Candidate(
            "compose",
            "row",
            {"operator_path": "sequential_composition"},
            admission_status="passed",
        ),
    )
    runtime = vp.composition_runtime_config(
        operator,
        order=("identity",),
        components={"identity": child.component},
        anchor_components={"identity": child.anchor_component},
        children=(child,),
        candidates=candidates,
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={"identity": "test.identity_component"},
        anchor_component_signature={"identity": "test.identity_component"},
        axis_registry=None,
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    names = tuple(record.name for record in plan.check_records)
    child_record = next(
        record for record in plan.check_records if record.name == "child_anchor"
    )
    parent_record = next(
        record for record in plan.check_records if record.name == "composition_anchor"
    )

    assert names == ("child_anchor", "composition_anchor")
    assert child_record.family == "child"
    assert parent_record.measurements["child_reference_owner_hashes"] == (
        child_record.owner_hash,
    )
    assert (
        tmp_path
        / "candidates"
        / child.candidate.family
        / child.candidate.candidate_id
        / f"{child.candidate.candidate_spec_hash()}.json"
    ).exists()


def test_composition_replay_requires_child_reference_rows(tmp_path: Path) -> None:
    model = OneParameterModule()
    operator = vp.composition("compose", "child_anchor", aggregation="none")
    child = vp.CompositionChild(
        name="identity",
        candidate=vp.Candidate(
            "child",
            "identity",
            {"operator_path": "child"},
            admission_status="passed",
        ),
        component=identity_component,
        anchor_component=identity_component,
        reference_check=passed_child_reference,
        input_signature={"child": "identity"},
    )
    parent = vp.Candidate(
        "compose",
        "row",
        {"operator_path": "sequential_composition"},
        admission_status="passed",
    )
    runtime = vp.composition_runtime_config(
        operator,
        order=("identity",),
        components={"identity": child.component},
        anchor_components={"identity": child.anchor_component},
        children=(child,),
        candidates=(parent,),
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={"identity": "test.identity_component"},
        anchor_component_signature={"identity": "test.identity_component"},
        axis_registry=None,
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    child_record = next(
        record for record in plan.check_records if record.name == "child_anchor"
    )
    parent_record = next(
        record for record in plan.check_records if record.name == "composition_anchor"
    )
    child_candidate_row = read_record(
        tmp_path
        / "candidates"
        / child.candidate.family
        / child.candidate.candidate_id
        / f"{child.candidate.candidate_spec_hash()}.json"
    )
    parent_candidate_row = read_record(
        tmp_path
        / "candidates"
        / parent.family
        / parent.candidate_id
        / f"{parent.candidate_spec_hash()}.json"
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")
    full_size_rows = tuple(
        vp.full_size_record_from_json(
            read_record(
                tmp_path
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
                tmp_path
                / "references"
                / record.family
                / record.candidate_id
                / record.candidate_spec_hash
                / f"{record.name}.json"
            )
        )
        for record in plan.check_records
    )
    child_record = next(
        record for record in check_rows if record.name == "child_anchor"
    )
    parent_record = next(
        record for record in check_rows if record.name == "composition_anchor"
    )

    replayed = vp.plan_from_json(
        summary,
        replay_context=replay_context_for_plan(plan),
        full_size_records=full_size_rows,
        check_records=check_rows,
        candidate_records=(parent_candidate_row, child_candidate_row),
        materializers=plan.materializers,
        run_dir=tmp_path,
    )

    assert tuple(record.name for record in replayed.check_records) == (
        "child_anchor",
        "composition_anchor",
    )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=full_size_rows,
            check_records=(parent_record,),
            candidate_records=(parent_candidate_row,),
            materializers=plan.materializers,
            run_dir=tmp_path,
        )

    failed_child = dataclasses.replace(child_record, status="failed")

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=full_size_rows,
            check_records=(failed_child, parent_record),
            candidate_records=(parent_candidate_row, child_candidate_row),
            materializers=plan.materializers,
            run_dir=tmp_path,
        )


def test_composition_reference_check_low_precision_anchor() -> None:
    check = vp.composition_reference_check(
        vp.composition("compose", "identity", aggregation="none"),
        order=("identity",),
        components={"identity": identity_component},
        anchor_components={"identity": identity_component},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vp.Candidate(
                "compose",
                "float32",
                {
                    "operator_path": "sequential_composition",
                    "model_dtype": "float32",
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.00000006], dtype=torch.float64)},
        )


def test_standard_runtime_tunes_hvp_and_materializes_selected_operator(
    tmp_path: Path,
) -> None:
    model = OneParameterModule()
    params = {"w": model.w.detach().clone()}
    candidates = (
        vp.Candidate(
            "hvp",
            "reverse",
            {"operator_path": "reverse_over_reverse"},
            changed_axes=("operator_path",),
        ),
        vp.Candidate(
            "hvp",
            "jvp-grad",
            {
                "operator_path": "jvp_grad",
                **torch_func_settings(requires_forward_ad=True),
            },
            changed_axes=("operator_path",),
        ),
    )
    operator = vp.hvp("hvp", "loss", aggregation="sum")
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=vp.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=candidates,
            thresholds={
                "max_abs_diff": 1e-9,
                "max_rel_diff": 1e-9,
                "directional_abs_diff": 1e-3,
                "directional_rel_diff": 1e-3,
            },
            objective_signature={"loss": "quadratic-v1"},
            axis_registry=vp.standard_axis_registry(),
            scalar_objectives={"loss": quadratic_scalar},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0, 2.0, 3.0)),
    )
    selected = vp.materialize(plan, family="hvp")
    result = selected(
        {"family": "hvp", "scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert plan.selected["hvp"].candidate_id == "jvp-grad"
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([12.0], dtype=torch.float64),
    )
