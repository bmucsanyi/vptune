import dataclasses
from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any

import pytest
import torch
from torch._dynamo import config as torch_dynamo_config

import vptune as vp
import vptune.ext as vpx
import vptune.runtime as runtime_module
from vptune import operators as ops
from vptune.io import read_record
from vptune.measure import CPUMemoryBackend
from vptune.run import tune as tune_problem
from vptune.tensor_tree import tree_leaves, tree_map


class OneParameterModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))


class TwoParameterModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))


class MatrixParameterModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TinyModelOutput:
    logits: torch.Tensor


class IdentifiedTeacherObjective:
    def __init__(self, version: str) -> None:
        self.version = version

    def __call__(
        self,
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert params
        assert buffers == {}
        assert batch
        assert context.family == "gradient"

        return {"logits": torch.tensor([1.0], dtype=torch.float64)}

    def identity(self) -> Mapping[str, object]:
        return {"version": self.version}


class RecordingTeacherObjective:
    def __init__(self, version: str, calls: list[object]) -> None:
        self.version = version
        self.calls = calls

    def __call__(
        self,
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert params["w"] is not None
        assert buffers == {}
        assert context.family == "gradient"
        self.calls.append(batch["family"])

        return {"logits": batch["teacher_seed"] + 1.0}

    def identity(self) -> Mapping[str, object]:
        return {"version": self.version}


class StatefulScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.register_buffer("b", torch.tensor([3.0], dtype=torch.float64))

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.w * scale + self.get_buffer("b")).sum()


class MutatingStatefulScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.register_buffer("b", torch.tensor([3.0], dtype=torch.float64))

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        self.get_buffer("b").add_(scale)

        return (self.w * self.get_buffer("b")).sum()


class DtypeObservingStatefulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.register_buffer("b", torch.tensor([3.0], dtype=torch.float64))
        self.observed = {}

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        self.observed["parameter"] = self.w.dtype
        self.observed["buffer"] = self.get_buffer("b").dtype
        self.observed["batch"] = scale.dtype

        return (self.w * scale + self.get_buffer("b")).sum()


class StatefulOutputModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
        self.register_buffer("b", torch.tensor([0.5, 1.5], dtype=torch.float64))

    def forward(self, scale: torch.Tensor) -> TinyModelOutput:
        return TinyModelOutput(logits=self.w * scale + self.get_buffer("b"))


class ScaleData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "scale"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {
            "family": family,
            "check": check_name,
            "scale": 2.0,
            "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
        }

    @staticmethod
    def probe_batches(family: str) -> Sequence[Mapping[str, object]]:
        return ({"family": family, "scale": 2.0},)


class TeacherOutputData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "teacher_outputs"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {
            "family": family,
            "check": check_name,
            "scale": 2.0,
            "teacher_seed": torch.tensor([2.0], dtype=torch.float64),
            "teacher_outputs": {"logits": torch.tensor([3.0], dtype=torch.float64)},
        }

    @staticmethod
    def probe_batches(family: str) -> Sequence[Mapping[str, object]]:
        return (
            {
                "family": family,
                "scale": 2.0,
                "teacher_seed": torch.tensor([2.0], dtype=torch.float64),
                "teacher_outputs": {"logits": torch.tensor([3.0], dtype=torch.float64)},
            },
        )


class ParameterVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "parameter_vector"}

    @staticmethod
    def reference_vectors(family: str) -> vpx.TensorTree:
        assert family

        return {"w": torch.tensor([3.0], dtype=torch.float64)}

    @staticmethod
    def probe_vectors(family: str) -> Sequence[vpx.TensorTree]:
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

        return {"metric_matrix": cls.matrix}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"metric_matrix": cls.matrix},)


class DiagonalMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "diagonal_metric"}

    @staticmethod
    def metric_diagonal() -> Mapping[str, torch.Tensor]:
        return {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"metric_diagonal": cls.metric_diagonal()}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"metric_diagonal": cls.metric_diagonal()},)


class BlockMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "block_metric"}

    @staticmethod
    def metric_blocks() -> tuple[torch.Tensor, ...]:
        return (torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64),)

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"metric_blocks": cls.metric_blocks()}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"metric_blocks": cls.metric_blocks()},)


class LowRankMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "low_rank_metric"}

    @staticmethod
    def factors() -> Mapping[str, torch.Tensor]:
        return {
            "basis": torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            "diagonal": torch.tensor([4.0, 5.0], dtype=torch.float64),
        }

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"low_rank_factors": cls.factors()}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"low_rank_factors": cls.factors()},)


class KFACMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "kfac_metric"}

    @staticmethod
    def factors() -> Mapping[str, torch.Tensor]:
        return {
            "w_left": torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64),
            "w_right": torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64),
        }

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"kfac_factors": cls.factors()}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"kfac_factors": cls.factors()},)


class EKFACMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "ekfac_metric"}

    @staticmethod
    def factors() -> Mapping[str, Mapping[str, torch.Tensor]]:
        scale = 2.0**-0.5

        return {
            "ekfac_eigvecs_a": {
                "w": torch.tensor(
                    [[scale, -scale], [scale, scale]],
                    dtype=torch.float64,
                )
            },
            "ekfac_eigvecs_g": {
                "w": torch.tensor(
                    [[0.8, -0.6], [0.6, 0.8]],
                    dtype=torch.float64,
                )
            },
            "ekfac_corrected_eigenvalues": {
                "w": torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.float64)
            },
        }

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return cls.factors()

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return (cls.factors(),)


class GGNMetricData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "ggn_metric"}

    @staticmethod
    def factors() -> Mapping[str, torch.Tensor]:
        return {
            "jacobian": torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64),
            "loss_hessian": torch.tensor(
                [[3.0, 0.25], [0.25, 2.0]],
                dtype=torch.float64,
            ),
        }

    @classmethod
    def reference_batch(
        cls,
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        assert family
        assert check_name

        return {"ggn_factors": cls.factors()}

    @classmethod
    def probe_batches(cls, family: str) -> Sequence[Mapping[str, object]]:
        assert family

        return ({"ggn_factors": cls.factors()},)


class TwoParameterVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_parameter_vector"}

    @staticmethod
    def reference_vectors(family: str) -> vpx.TensorTree:
        assert family

        return {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    @staticmethod
    def probe_vectors(family: str) -> Sequence[vpx.TensorTree]:
        assert family

        return ({"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},)


def matrix_vector_tensor() -> torch.Tensor:
    return torch.tensor(
        [[0.25, -0.75], [0.5, 1.25]],
        dtype=torch.float64,
    )


class MatrixVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "matrix_vector"}

    @staticmethod
    def reference_vectors(family: str) -> vpx.TensorTree:
        assert family

        return {"w": matrix_vector_tensor()}

    @staticmethod
    def probe_vectors(family: str) -> Sequence[vpx.TensorTree]:
        assert family

        return ({"w": matrix_vector_tensor()},)


def passed_candidate(
    family: str,
    candidate_id: str,
    settings: Mapping[str, Any],
) -> vpx.Candidate:
    return vpx.Candidate(
        family,
        candidate_id,
        dict(settings),
        admission_status="passed",
    )


def run_passed_candidate(
    factory: Callable[
        [vpx.Candidate, vpx.Batch, vpx.TensorTree],
        vpx.CandidateOperation,
    ],
    family: str,
    candidate_id: str,
    settings: Mapping[str, Any],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.TensorTree:
    return factory(
        passed_candidate(family, candidate_id, settings),
        batch,
        vector,
    )()


def candidate_leaf_results(
    factory: Any,
    family: str,
    cases: Sequence[tuple[str, Mapping[str, Any]]],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> dict[str, torch.Tensor]:
    return {
        candidate_id: tree_leaves(
            run_passed_candidate(
                factory,
                family,
                candidate_id,
                settings,
                batch,
                vector,
            )
        )[0]
        for candidate_id, settings in cases
    }


def standard_factory(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    operator: vpx.OperatorSpec,
    **kwargs: Any,
) -> Any:
    return vpx.standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        **kwargs,
    )


def quadratic_hvp_factory(
    params: vpx.ParameterTree,
) -> Callable[[vpx.Candidate, vpx.Batch, vpx.TensorTree], vpx.CandidateOperation]:
    return vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )


def run_quadratic_hvp(
    candidate_id: str,
    settings: Mapping[str, Any],
    params: vpx.ParameterTree,
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.TensorTree:
    return run_passed_candidate(
        quadratic_hvp_factory(params),
        "hvp",
        candidate_id,
        settings,
        batch,
        vector,
    )


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1

        return value


def cpu_target() -> vpx.Target:
    return vpx.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float64", "float32", "bfloat16", "float16"),
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=vpx.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        ),
        selection_policy=vpx.SelectionPolicy(),
        search_policy=vpx.SearchPolicy(strategy="exhaustive"),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def replay_context_for_plan(plan: vpx.Plan) -> vpx.ReplayContext:
    family_input_signatures = {
        family: record.input_signature for family, record in plan.records.items()
    }

    for record in plan.check_records:
        family_input_signatures.setdefault(record.family, record.input_signature)

    return vpx.ReplayContext(
        input_signature=plan.input_signature,
        family_input_signatures=family_input_signatures,
        materializer_identities=plan.materializer_identities(),
        selection_policy=plan.policy,
        target_identity=plan.target_identity,
        runtime_identities=plan.selected_runtime_identities(),
        adapter_identities=plan.selected_adapter_identities(),
        validator_identities=plan.selected_validator_identities(),
    )


def quadratic_scalar(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    assert buffers == {}
    assert context.family

    return batch["scale"] * params["w"].pow(2).sum()


def mutating_buffer_scalar(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    assert batch == {}
    assert context.family == "gradient"
    buffers["b"].add_(1.0)

    return (params["w"] * buffers["b"]).sum()


def failing_mutating_buffer_scalar(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    assert params["w"] is not None
    assert batch == {}
    assert context.family == "gradient"
    buffers["b"].add_(1.0)
    message = "declared mutation failure"

    raise RuntimeError(message)


def declared_restored_functional_call_settings(
    *,
    mutates_state: bool = True,
    mutated_parameter_keys: tuple[str, ...] = (),
    mutated_buffer_keys: tuple[str, ...] = ("b",),
) -> dict[str, object]:
    return {
        "call.path": "functional_call",
        "call.params": "explicit_params",
        "call.buffers": "explicit_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "declared_and_restored",
        "call.grad_mode": "grad_enabled",
        "call.return_type": "raw_tensor_tree",
        "parameter_keys": ("w",),
        "buffer_keys": ("b",),
        "tie_weights": True,
        "strict": False,
        "parametrization_policy": "active",
        "mutates_state": mutates_state,
        "mutated_parameter_keys": mutated_parameter_keys,
        "mutated_buffer_keys": mutated_buffer_keys,
        "module_mode": "eval",
    }


def square_function(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> vpx.TensorTree:
    assert buffers == {}
    assert batch["scale"]
    assert context.family

    return {"y": params["w"].pow(2)}


def scaled_tree_function(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> vpx.TensorTree:
    assert buffers == {}
    assert context.family in {"jvp", "vjp"}

    return {"w": params["w"] * batch["scale"]}


def square_tensor_function(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    assert buffers == {}
    assert batch["loss_hessian"] is not None
    assert context.family

    return params["w"].pow(2)


def multiply_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    scale = batch["scale"]
    assert isinstance(scale, float)

    return tree_map(lambda tensor: tensor * scale, vector)


def shift_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 1.0, vector)


def wrong_shift_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 2.0, vector)


def identity_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    assert batch["scale"]

    return vector


def fused_multiply_shift_component(
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.TensorTree:
    return shift_component(batch, multiply_component(batch, vector))


def torch_func_settings(*, requires_forward_ad: bool) -> dict[str, object]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def gradient_settings() -> dict[str, object]:
    return {"gradient.path": "torch_autograd_grad"}


def standard_checkpoint_settings() -> dict[str, object]:
    return {
        "activation.recompute": "checkpoint_non_reentrant_by_layer",
        "activation.offload": "none",
        "checkpoint.use_reentrant": "false",
        "checkpoint.early_stop": "true",
        "checkpoint.preserve_rng_state": "true",
        "checkpoint.determinism_check": "default",
        "checkpoint.context_fn": "none",
        "checkpoint.moves_to_new_device": "false",
        "checkpoint.uses_global_state": "false",
    }


def jvp_settings(path: str) -> dict[str, object]:
    return {"jvp.path": path}


def vjp_settings() -> dict[str, object]:
    return {"vjp.path": "torch_func_vjp"}


def metric_settings(path: str = "dense_matmul") -> dict[str, object]:
    if path == "dense_matmul":
        return {"metric.multiply_path": path}

    if path == "streaming_multiply":
        return {"metric.multiply_path": path, "metric.accumulation": "streaming"}

    return {
        "metric.multiply_path": path,
        "metric.accumulation": "materialized_blocks",
    }


def dense_metric_representation() -> dict[str, object]:
    return {"kind": "dense_matrix"}


def diagonal_metric_representation() -> dict[str, object]:
    return {"kind": "diagonal_tree"}


def block_metric_representation() -> dict[str, object]:
    return {"kind": "block_diagonal"}


def low_rank_metric_representation() -> dict[str, object]:
    return {"kind": "low_rank_factors"}


def kfac_metric_representation() -> dict[str, object]:
    return {
        "kind": "kfac_factors",
        "blocks": (
            {
                "parameter": "w",
                "left_factor": "w_left",
                "right_factor": "w_right",
            },
        ),
    }


def ekfac_metric_representation() -> dict[str, object]:
    return {"kind": "ekfac_factors"}


def ekfac_dense_matrix(
    factors: Mapping[str, Mapping[str, torch.Tensor]],
    key: str = "w",
) -> torch.Tensor:
    eigvecs_a = factors["ekfac_eigvecs_a"][key]
    eigvecs_g = factors["ekfac_eigvecs_g"][key]
    eigenvalues = factors["ekfac_corrected_eigenvalues"][key]
    basis = torch.kron(eigvecs_a, eigvecs_g)

    return basis @ torch.diag(eigenvalues.reshape(-1)) @ basis.T


def ekfac_square_root_product(
    factors: Mapping[str, Mapping[str, torch.Tensor]],
    vector: torch.Tensor,
    *,
    inverse: bool,
    damping: float = 0.0,
    key: str = "w",
) -> torch.Tensor:
    eigvecs_a = factors["ekfac_eigvecs_a"][key]
    eigvecs_g = factors["ekfac_eigvecs_g"][key]
    eigenvalues = factors["ekfac_corrected_eigenvalues"][key]
    spectrum = eigenvalues + damping if inverse else eigenvalues
    coeffs = eigvecs_a.T @ vector @ eigvecs_g
    factors_matrix = torch.rsqrt(spectrum) if inverse else torch.sqrt(spectrum)

    return eigvecs_a @ (factors_matrix * coeffs) @ eigvecs_g.T


def symmetric_square_root(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    return eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T


def ggn_metric_representation() -> dict[str, object]:
    return {"kind": "ggn_derived_factors"}


def inverse_metric_settings(path: str = "dense_solve") -> dict[str, object]:
    return {"inverse_metric.solve_path": path}


def hvp_settings(path: str) -> dict[str, object]:
    return {"hvp.path": path}


def hvp_gradient_reuse_settings() -> dict[str, object]:
    return {
        **hvp_settings("linearize_grad"),
        **torch_func_settings(requires_forward_ad=False),
        "hvp.graph_schedule": "rebuild_graph_per_vector",
        "hvp.primal_reuse": "recompute_primal",
        "hvp.gradient_reuse": "reuse_gradient_closure",
    }


def ggn_settings(
    path: str,
    *,
    vjp_path: str = "torch_func_vjp",
) -> dict[str, object]:
    return {"ggn.jvp_path": path, "ggn.vjp_path": vjp_path}


def ggn_dense_kernel_settings() -> dict[str, object]:
    return {
        **ggn_settings("torch_func_jvp"),
        **torch_func_settings(requires_forward_ad=True),
        "ggn.loss_hessian_path": "autodiff_loss_hvp",
        "ggn.loss_hessian_kernel": "dense_global",
    }


def ggn_closed_form_ce_kl_settings(kernel: str) -> dict[str, object]:
    settings = {
        **ggn_settings("torch_func_jvp"),
        **torch_func_settings(requires_forward_ad=True),
        "ggn.loss_hessian_path": "closed_form_softmax_ce_kl",
        "ggn.loss_hessian_kernel": kernel,
    }

    if kernel == "two_pass_chunked_global":
        settings["chunk.class_block_size_with_exact_global_normalization"] = 2

    return settings


def ggn_reuse_settings(jvp_reuse: str, cotangent_reuse: str) -> dict[str, object]:
    return {
        **ggn_dense_kernel_settings(),
        "ggn.jvp_reuse": jvp_reuse,
        "ggn.cotangent_reuse": cotangent_reuse,
    }


def dense_softmax_ce_kl_loss_hessian(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    blocks = [
        torch.diag(row) - torch.outer(row, row)
        for row in probabilities.reshape(-1, probabilities.shape[-1])
    ]

    return torch.block_diag(*blocks)


def fisher_settings(
    accumulation: str,
    *,
    score_grad_path: str = "torch_autograd_grad_loop",
) -> dict[str, object]:
    if accumulation == "streaming_dot_accumulate":
        return {
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": accumulation,
            "fisher.score_grad_path": score_grad_path,
        }

    return {
        "fisher.expectation_path": "explicit_full_expectation_score_rows",
        "fisher.accumulation": accumulation,
    }


def sampled_fisher_settings(
    accumulation: str,
    *,
    sample_source: str = "fixed_seed_and_count",
    exact_check: str = "disabled",
) -> dict[str, object]:
    return {
        "sampled_fisher.accumulation": accumulation,
        "sampled_fisher.sample_source": sample_source,
        "sampled_fisher.exact_fisher_check": exact_check,
    }


def sampled_fisher_grad_settings(path: str) -> dict[str, object]:
    return {
        **sampled_fisher_settings("streaming_dot_accumulate"),
        "sampled_fisher.score_grad_path": path,
    }


def empirical_grad_settings(path: str) -> dict[str, object]:
    return {"empirical_fisher.grad_path": path}


def vmap_settings(
    in_dims: dict[str, int | None],
    *,
    chunk_size: int = 1,
) -> dict[str, object]:
    return {
        "vectorization.mode": "vmap",
        "vectorization.vmap_chunk_size": chunk_size,
        "vectorization.in_dims": in_dims,
    }


def fisher_per_example_vmap_settings(*, batch_size: int = 1) -> dict[str, object]:
    return {
        "schedule.per_example": "vmap",
        "batch.fisher_sample_batch_size": batch_size,
    }


def fisher_per_example_manual_settings(*, batch_size: int = 1) -> dict[str, object]:
    return {
        "schedule.per_example": "manual_batch",
        "batch.fisher_sample_batch_size": batch_size,
    }


def empirical_per_example_vmap_settings(*, batch_size: int = 1) -> dict[str, object]:
    return {
        "schedule.per_example": "vmap",
        "batch.empirical_example_batch_size": batch_size,
    }


def empirical_per_example_manual_settings(*, batch_size: int = 1) -> dict[str, object]:
    return {
        "schedule.per_example": "manual_batch",
        "batch.empirical_example_batch_size": batch_size,
    }


def empirical_dense_settings() -> dict[str, object]:
    return {"empirical_fisher.accumulation": "materialize_per_example_gradients"}


def per_example_gradient_settings(
    *,
    accumulation: str = "stacked_leading_axis",
    block_size: int | None = None,
) -> dict[str, object]:
    if block_size is None:
        return {
            "per_example_gradient.grad_path": "torch_autograd_grad_loop",
            "per_example_gradient.accumulation": accumulation,
        }

    return {
        "per_example_gradient.grad_path": "torch_autograd_grad_loop",
        "per_example_gradient.accumulation": accumulation,
        "batch.per_example_block_size": block_size,
    }


def composition_settings(
    *,
    execution: str = "stream_child_outputs",
    child_evaluation: str = "inline_child_lowering",
    validation: str = "validate_composed_output",
) -> dict[str, object]:
    return {
        "composition.execution": execution,
        "composition.child_evaluation": child_evaluation,
        "composition.validation": validation,
    }


def compile_settings(
    *,
    boundary: str = "whole_operator",
    cache_state: str = "cold_compile",
    mode: str | None = "default",
    cuda_graphs: str = "false",
    compiled_autograd: str = "false",
) -> dict[str, object]:
    return {
        "compile.enabled": "true",
        "compile.boundary": boundary,
        "compile.backend": "inductor",
        "compile.mode": mode,
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": compiled_autograd,
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": cuda_graphs,
        "compile.cache_state": cache_state,
    }


class CompileRecorder:
    def __init__(
        self,
        events: list[Any],
        *,
        tag: str | None = None,
        called_event: str | None = None,
        expected_mode: str | None = "default",
        expected_options: Mapping[str, object] | None = None,
        compile_event: Callable[[], Any] | None = None,
        call_event: Callable[..., Any] | None = None,
    ) -> None:
        self.events = events
        self.tag = tag
        self.called_event = called_event
        self.expected_mode = expected_mode
        self.expected_options = expected_options
        self.compile_event = compile_event
        self.call_event = call_event
        self.stack = []

    def active(self) -> bool:
        return len(self.stack) > 0

    def event(self, name: str) -> tuple[Any, ...]:
        if self.tag is None:
            return (name, self.active())

        return (name, self.tag, self.active())

    def compile(
        self,
        operation: Callable[..., Any],
        *,
        backend: str,
        mode: str | None,
        fullgraph: bool,
        dynamic: bool | None,
        options: Mapping[str, object] | None,
    ) -> Callable[..., Any]:
        assert backend == "inductor"
        assert mode == self.expected_mode
        assert fullgraph is False
        assert dynamic is None
        assert options == self.expected_options

        if self.compile_event is None:
            self.events.append(self.event("compile"))
        else:
            self.events.append(self.compile_event())

        def compiled(*args: Any, **kwargs: Any) -> Any:
            self.stack.append(True)

            if self.call_event is not None:
                self.events.append(self.call_event(*args, **kwargs))
            elif self.called_event is not None:
                self.events.append(self.event(self.called_event))

            try:
                return operation(*args, **kwargs)
            finally:
                self.stack.pop()

        return compiled


def patch_runtime_call_recorder(
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    name: str,
    recorder: CompileRecorder,
    event_name: str,
    event_factory: Callable[[tuple[Any, ...], dict[str, Any]], Any] | None = None,
) -> None:
    original = getattr(target, name)

    def recording(*args: Any, **kwargs: Any) -> Any:
        if event_factory is None:
            recorder.events.append(recorder.event(event_name))
        else:
            recorder.events.append(event_factory(args, kwargs))

        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, recording)


def patch_runtime_output_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module,
        "_runtime_output",
        recorder,
        "runtime_output",
    )


def patch_require_finite_tree_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module,
        "_require_finite_tree",
        recorder,
        "finite_tree",
        lambda args, _: ("finite_tree", args[1], recorder.active()),
    )


def patch_autograd_grad_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module.torch.autograd,
        "grad",
        recorder,
        "grad",
    )


def patch_ggn_loss_product_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module,
        "_ggn_loss_hessian_product",
        recorder,
        "loss_hessian",
    )


def patch_ggn_vjp_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module,
        "_run_ggnvp_vjp",
        recorder,
        "vjp",
    )


def patch_score_matrix_product_recorder(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CompileRecorder,
) -> None:
    patch_runtime_call_recorder(
        monkeypatch,
        runtime_module,
        "_score_matrix_product",
        recorder,
        "score_matrix_product",
    )


def patch_matmul_call_recorder(
    monkeypatch: pytest.MonkeyPatch,
    setting_key: str,
    setting_value: object,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    calls = []
    original_matmul = runtime_module._matmul_runtime

    def recording_matmul(
        settings: Mapping[str, Any],
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        if settings.get(setting_key) == setting_value:
            calls.append((tuple(left.shape), tuple(right.shape)))

        return original_matmul(settings, left, right)

    monkeypatch.setattr(runtime_module, "_matmul_runtime", recording_matmul)

    return calls


def compiled_boundary_operation(
    operator: vpx.OperatorSpec,
    *,
    params: vpx.ParameterTree,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    scalar_objectives: Mapping[str, Callable[..., torch.Tensor]] | None = None,
    function_objectives: Mapping[str, Callable[..., vpx.TensorTree]] | None = None,
) -> vpx.CandidateOperation:
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )

    return factory(
        passed_candidate(
            operator.family,
            "compiled",
            settings,
        ),
        batch,
        vector,
    )


def compiled_ggn_boundary_operation(
    settings: Mapping[str, object],
) -> vpx.CandidateOperation:
    return compiled_boundary_operation(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        settings=settings,
        batch={
            "scale": 1.0,
            "loss_hessian": torch.tensor([[5.0]], dtype=torch.float64),
        },
        vector={"w": torch.tensor([3.0], dtype=torch.float64)},
        function_objectives={"model_output": square_function},
    )


def assert_compiled_boundary_result(
    operator: vpx.OperatorSpec,
    *,
    params: vpx.ParameterTree,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    expected: torch.Tensor,
    scalar_objectives: Mapping[str, Callable[..., torch.Tensor]] | None = None,
    function_objectives: Mapping[str, Callable[..., vpx.TensorTree]] | None = None,
) -> None:
    operation = compiled_boundary_operation(
        operator,
        params=params,
        settings=settings,
        batch=batch,
        vector=vector,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )

    torch.testing.assert_close(tree_leaves(operation())[0], expected)


def compiled_metric_operation(
    settings: Mapping[str, object],
    *,
    params: vpx.ParameterTree,
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.CandidateOperation:
    return compiled_boundary_operation(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        settings=settings,
        batch=batch,
        vector=vector,
    )


def compiled_stateful_model_forward_operation(
    settings: Mapping[str, object],
) -> vpx.CandidateOperation:
    module = StatefulScalarModule()
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=dict(module.named_buffers()),
        module=module,
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )

    return factory(
        passed_candidate(
            "gradient",
            "compiled-model-forward",
            {
                **gradient_settings(),
                **stateful_module_call_settings(),
                **settings,
            },
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )


def test_per_example_gradient_stacked_and_blockwise_execute_declared_rows() -> None:
    params = {"w": torch.tensor([1.0, -1.0], dtype=torch.float64)}
    batch = {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
    expected = torch.tensor(
        [[1.0, 1.0], [2.0, 4.0], [3.0, 9.0]],
        dtype=torch.float64,
    )

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "per_example"
        x = batch["x"].reshape(-1)

        return x * params["w"][0] + x.square() * params["w"][1]

    factory = vpx.standard_operation_factory(
        ops.per_example_gradient(
            "per_example",
            "scores",
            aggregation="sum",
            example_loss_reduction="per_example",
        ),
        params=params,
        buffers={},
        function_objectives={"scores": score_rows},
    )
    registry = vpx.standard_axis_registry()

    for candidate_id, settings in (
        ("stacked", per_example_gradient_settings()),
        (
            "blockwise",
            per_example_gradient_settings(
                accumulation="blockwise_stacked",
                block_size=2,
            ),
        ),
    ):
        candidate = registry.admit(vpx.Candidate("per_example", candidate_id, settings))
        assert candidate.admission_status == "passed"
        output = factory(candidate, batch, {})()

        torch.testing.assert_close(tensor_mapping(output)["w"], expected)


def test_per_example_gradient_reference_check_and_empirical_outer_product() -> None:
    params = {"w": torch.tensor([1.0, -1.0], dtype=torch.float64)}
    batch = {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.5], dtype=torch.float64)}

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "per_example"
        x = batch["x"].reshape(-1)

        return x * params["w"][0] + x.square() * params["w"][1]

    operator = ops.per_example_gradient(
        "per_example",
        "scores",
        aggregation="sum",
        example_loss_reduction="per_example",
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
        function_objectives={"scores": score_rows},
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        function_objectives={"scores": score_rows},
    )
    registry = vpx.standard_axis_registry()
    stacked = registry.admit(
        vpx.Candidate(
            "per_example",
            "stacked",
            per_example_gradient_settings(),
        )
    )
    blockwise = registry.admit(
        vpx.Candidate(
            "per_example",
            "blockwise",
            per_example_gradient_settings(
                accumulation="blockwise_stacked",
                block_size=2,
            ),
        )
    )

    assert stacked.admission_status == "passed"
    assert blockwise.admission_status == "passed"

    for candidate in (stacked, blockwise):
        result = check(candidate, batch, {})

        assert result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert result.measurements["max_rel_diff"] == pytest.approx(0.0)

    rows = tensor_mapping(factory(blockwise, batch, {})())["w"]
    empirical_factory = vpx.standard_operation_factory(
        ops.empirical_fisher_vp(
            "empirical",
            "scores",
            aggregation="mean_per_example",
            example_loss_reduction="per_example",
            denominator="batch_normalization",
        ),
        params=params,
        buffers={},
    )
    empirical_output = empirical_factory(
        vpx.Candidate(
            "empirical",
            "outer-product",
            empirical_dense_settings(),
            admission_status="passed",
        ),
        {"per_example_gradients": rows, "normalization": 3.0},
        vector,
    )()
    expected = rows.T @ (rows @ vector["w"]) / 3.0

    torch.testing.assert_close(tensor_mapping(empirical_output)["w"], expected)


def test_per_example_gradient_block_size_requires_blockwise_accumulation() -> None:
    candidate = vpx.standard_axis_registry().admit(
        vpx.Candidate(
            "per_example",
            "bad-block-size",
            per_example_gradient_settings(block_size=2),
        )
    )

    assert candidate.admission_status == "failed"
    assert "blockwise_stacked" in str(candidate.admission_error)


def test_compile_backend_accepts_concrete_registered_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.torch.compiler,
        "list_backends",
        lambda: ["custom_backend"],
    )

    assert (
        runtime_module._compile_backend({"compile.backend": "custom_backend"})
        == "custom_backend"
    )


def test_compile_backend_requires_backend_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(runtime_module.torch.compiler, "list_backends")

    with pytest.raises(vp.MaterializationError, match="list_backends"):
        runtime_module._compile_backend({"compile.backend": "custom_backend"})


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ("registered_backend", "concrete PyTorch compiler backend id"),
        ("missing_backend", "not registered"),
    ],
)
def test_compile_backend_rejects_non_concrete_or_unregistered_backend(
    backend: str,
    message: str,
) -> None:
    with pytest.raises(vp.MaterializationError, match=message):
        runtime_module._compile_backend({"compile.backend": backend})


def loss_scaling_settings(*, degree: int, scale: float = 8.0) -> dict[str, object]:
    return {
        "numeric.loss_scaling": "static_scale_with_exact_unscale",
        "numeric.loss_scale": scale,
        "numeric.loss_unscale_degree": degree,
    }


def numeric_bound_fields(
    *,
    k: int = 1,
    epsilon: float = 1e-8,
    c_op: float = 1.0,
    s_row: float = 1.0,
    output_norm_floor: float = 1e-6,
) -> dict[str, object]:
    return {
        "k": k,
        "epsilon": epsilon,
        "C_op": c_op,
        "S_row": s_row,
        "output_norm_floor": output_norm_floor,
    }


def score_terms_fisher(family: str, objective_id: str) -> vpx.OperatorSpec:
    return ops.fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="explicit_scores",
        sample_space="terms",
        score_reduction="none",
        denominator="batch_normalization",
    )


def valid_sampling_bound() -> dict[str, object]:
    return {
        "kind": "abs_or_rel",
        "max_abs_diff": 1e-12,
        "max_rel_diff": 1e-12,
        "norm_floor": 1e-12,
    }


def score_terms_sampled_fisher(
    family: str,
    objective_id: str,
    *,
    sample_source: str = "fixed_seed_and_count",
    sampling_bound: Mapping[str, object] | None = None,
) -> vpx.OperatorSpec:
    bound = valid_sampling_bound() if sampling_bound is None else sampling_bound

    return ops.sampled_fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="sampled_labels",
        sample_count=2,
        sample_source=sample_source,
        sampling_bound=bound,
        score_reduction="none",
        denominator="num_examples",
    )


def score_terms_fisher_sum(family: str, objective_id: str) -> vpx.OperatorSpec:
    return ops.fisher_vp(
        family,
        objective_id,
        aggregation="sum",
        distribution="explicit_score_gradients",
        label_policy="explicit_scores",
        sample_space="terms",
        score_reduction="none",
        denominator="one",
    )


def score_terms_sampled_fisher_sum(
    family: str,
    objective_id: str,
    *,
    sample_source: str = "fixed_seed_and_count",
) -> vpx.OperatorSpec:
    return ops.sampled_fisher_vp(
        family,
        objective_id,
        aggregation="sum",
        distribution="explicit_score_gradients",
        label_policy="sampled_labels",
        sample_count=2,
        sample_source=sample_source,
        sampling_bound=valid_sampling_bound(),
        score_reduction="none",
        denominator="one",
    )


def empirical_fisher_sum(family: str, objective_id: str) -> vpx.OperatorSpec:
    return ops.empirical_fisher_vp(
        family,
        objective_id,
        aggregation="sum",
        example_loss_reduction="per_example",
        denominator="one",
    )


def empirical_fisher_mean(
    family: str,
    objective_id: str,
    *,
    denominator: str = "num_examples",
) -> vpx.OperatorSpec:
    return ops.empirical_fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        example_loss_reduction="per_example",
        denominator=denominator,
    )


def empirical_loss_factory(
    params: vpx.ParameterTree,
    function_objective: vpx.FunctionObjective,
) -> Callable[[vpx.Candidate, vpx.Batch, vpx.TensorTree], vpx.CandidateOperation]:
    return vpx.standard_operation_factory(
        empirical_fisher_mean("empirical", "losses"),
        params=params,
        buffers={},
        function_objectives={"losses": function_objective},
    )


def fisher_family_score_factories(
    params: vpx.ParameterTree,
    function_objective: vpx.FunctionObjective,
) -> dict[str, Any]:
    return {
        "fisher": vpx.standard_operation_factory(
            score_terms_fisher("fisher", "scores"),
            params=params,
            buffers={},
            function_objectives={"scores": function_objective},
        ),
        "sampled": vpx.standard_operation_factory(
            score_terms_sampled_fisher("sampled", "scores"),
            params=params,
            buffers={},
            function_objectives={"scores": function_objective},
        ),
        "empirical": vpx.standard_operation_factory(
            empirical_fisher_mean("empirical", "scores"),
            params=params,
            buffers={},
            function_objectives={"scores": function_objective},
        ),
    }


def linear_score_rows(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    assert buffers == {}
    assert context.family in {"fisher", "sampled", "empirical", "per_example"}
    x = batch["x"].reshape(-1)

    return x * params["w"][0] + 2.0 * x * params["w"][1]


def add_one_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor + 1.0, vector)


def subtract_one_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
    assert batch["scale"]

    return tree_map(lambda tensor: tensor - 1.0, vector)


def passed_child_reference(
    candidate: vpx.Candidate,
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.ReferenceResult:
    assert candidate.family == "child"
    assert batch["scale"]
    assert vector

    return vpx.ReferenceResult("child_anchor", {}, {"child_value": 0.0})


def failed_child_reference(
    candidate: vpx.Candidate,
    batch: vpx.Batch,
    vector: vpx.TensorTree,
) -> vpx.ReferenceResult:
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
    factories = {
        "gradient": standard_factory(
            params,
            buffers,
            ops.gradient("gradient", "loss", aggregation="sum"),
            scalar_objectives={"loss": quadratic_scalar},
        ),
        "jvp": standard_factory(
            params,
            buffers,
            ops.jvp("jvp", "function", aggregation="none"),
            function_objectives={"function": square_function},
        ),
        "vjp": standard_factory(
            params,
            buffers,
            ops.vjp("vjp", "function", aggregation="none"),
            function_objectives={"function": square_function},
        ),
        "hvp": standard_factory(
            params,
            buffers,
            ops.hvp("hvp", "loss", aggregation="sum"),
            scalar_objectives={"loss": quadratic_scalar},
        ),
    }

    def assert_product(
        family: str,
        settings: Mapping[str, object],
        product_vector: vpx.TensorTree,
        expected_value: float,
    ) -> None:
        result = run_passed_candidate(
            factories[family],
            family,
            "row",
            settings,
            batch,
            product_vector,
        )

        torch.testing.assert_close(
            tree_leaves(result)[0],
            torch.tensor([expected_value], dtype=torch.float64),
        )

    cases = (
        ("gradient", {"gradient.path": "torch_autograd_grad"}, vector, 20.0),
        (
            "gradient",
            {
                "gradient.path": "torch_func_grad",
                **torch_func_settings(requires_forward_ad=False),
            },
            vector,
            20.0,
        ),
        (
            "gradient",
            {
                "gradient.path": "torch_func_grad_and_value",
                **torch_func_settings(requires_forward_ad=False),
            },
            vector,
            20.0,
        ),
        (
            "gradient",
            {"gradient.path": "backward_materialized_grad"},
            vector,
            20.0,
        ),
        (
            "jvp",
            {
                "jvp.path": "torch_func_jvp",
                **torch_func_settings(requires_forward_ad=True),
            },
            vector,
            12.0,
        ),
        (
            "jvp",
            {
                "jvp.path": "forward_ad_dual",
                "requires_forward_ad": True,
                "forward_ad_supported": True,
            },
            vector,
            12.0,
        ),
        (
            "jvp",
            {
                "jvp.path": "torch_func_linearize",
                **torch_func_settings(requires_forward_ad=True),
            },
            vector,
            12.0,
        ),
        (
            "vjp",
            {
                "vjp.path": "torch_func_vjp",
                **torch_func_settings(requires_forward_ad=False),
            },
            cotangent,
            16.0,
        ),
        ("vjp", {"vjp.path": "autograd_grad_outputs"}, cotangent, 16.0),
        ("vjp", {"vjp.path": "backward_materialized_grad"}, cotangent, 16.0),
        (
            "hvp",
            {
                **hvp_settings("jvp_grad"),
                **torch_func_settings(requires_forward_ad=True),
            },
            vector,
            30.0,
        ),
        (
            "hvp",
            {
                **hvp_settings("linearize_grad"),
                **torch_func_settings(requires_forward_ad=True),
            },
            vector,
            30.0,
        ),
        (
            "hvp",
            {
                **hvp_settings("forward_ad_dual"),
                "requires_forward_ad": True,
                "forward_ad_supported": True,
            },
            vector,
            30.0,
        ),
        ("hvp", hvp_settings("autograd_functional_hvp"), vector, 30.0),
        ("hvp", hvp_settings("autograd_functional_vhp"), vector, 30.0),
    )

    for family, settings, product_vector, expected_value in cases:
        assert_product(family, settings, product_vector, expected_value)


def test_numeric_loss_scaling_scales_gradient_source_and_unscales_output() -> None:
    backward_scales = []
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    batch = {"scale": 1.0}
    vector = {"w": torch.tensor([0.0], dtype=torch.float64)}

    class RecordedSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(value)

            return value.pow(2).sum()

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            (value,) = ctx.saved_tensors
            grad_output = grad_outputs[0]
            backward_scales.append(float(grad_output.detach()))

            return (grad_output * 2.0 * value,)

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"] == pytest.approx(1.0)
        assert context.family == "gradient"

        return RecordedSquare.apply(params["w"])

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "scaled",
            {
                "gradient.path": "torch_autograd_grad",
                **loss_scaling_settings(degree=1, scale=8.0),
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    assert backward_scales == [pytest.approx(8.0)]
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([4.0], dtype=torch.float64),
    )


def test_numeric_loss_scaling_rejects_wrong_operator_degree() -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="does not match operator"):
        factory(
            vpx.Candidate(
                "gradient",
                "wrong-degree",
                {
                    "gradient.path": "torch_autograd_grad",
                    **loss_scaling_settings(degree=2),
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )()


def test_gradient_value_reuse_axis_requires_value_and_gradient_path() -> None:
    registry = vpx.standard_axis_registry()
    admitted = registry.admit(
        vpx.Candidate(
            "gradient",
            "good",
            {
                "gradient.path": "torch_func_grad_and_value",
                "gradient.value_reuse": "gradient_and_primal_value",
                **torch_func_settings(requires_forward_ad=False),
            },
        )
    )
    rejected = registry.admit(
        vpx.Candidate(
            "gradient",
            "bad",
            {
                "gradient.path": "torch_func_grad",
                "gradient.value_reuse": "gradient_and_primal_value",
                **torch_func_settings(requires_forward_ad=False),
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"
    assert rejected.admission_error == (
        "gradient_and_primal_value requires torch_func_grad_and_value"
    )


def test_gradient_value_reuse_executes_value_and_gradient_path() -> None:
    calls = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"] == pytest.approx(1.0)
        assert context.family == "gradient"
        calls.append("scalar")

        return params["w"].pow(2).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([3.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "reuse-value",
            {
                "gradient.path": "torch_func_grad_and_value",
                "gradient.value_reuse": "gradient_and_primal_value",
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()

    assert calls == ["scalar"]
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([6.0], dtype=torch.float64),
    )


def test_gradient_value_reuse_rejects_nonfinite_primal_value() -> None:
    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"] == pytest.approx(1.0)
        assert context.family == "gradient"

        return params["w"].sum() + torch.tensor(float("nan"), dtype=torch.float64)

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([3.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": scalar},
    )

    with pytest.raises(vp.MaterializationError, match="gradient primal value"):
        factory(
            vpx.Candidate(
                "gradient",
                "reuse-value",
                {
                    "gradient.path": "torch_func_grad_and_value",
                    "gradient.value_reuse": "gradient_and_primal_value",
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )()


def test_jvp_linearize_reuse_axis_requires_linearize_path() -> None:
    registry = vpx.standard_axis_registry()
    admitted = registry.admit(
        vpx.Candidate(
            "jvp",
            "good",
            {
                "jvp.path": "torch_func_linearize",
                "jvp.linearize_reuse": "reuse_at_same_primal",
                **torch_func_settings(requires_forward_ad=True),
            },
        )
    )
    rejected = registry.admit(
        vpx.Candidate(
            "jvp",
            "bad",
            {
                "jvp.path": "torch_func_jvp",
                "jvp.linearize_reuse": "reuse_at_same_primal",
                **torch_func_settings(requires_forward_ad=True),
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"
    assert (
        rejected.admission_error == "reuse_at_same_primal requires torch_func_linearize"
    )


def jvp_linearize_reuse_result(
    monkeypatch: pytest.MonkeyPatch,
    reuse: str,
    candidate_id: str,
) -> tuple[list[str], vpx.TensorTree, vpx.TensorTree]:
    events = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert buffers == {}
        assert batch["scale"] == pytest.approx(1.0)
        assert context.family == "jvp"
        events.append("function")

        return {"y": params["w"] * 2.0}

    def fake_linearize(
        tensor_function: Callable[[vpx.ParameterTree], vpx.TensorTree],
        params: vpx.ParameterTree,
    ) -> tuple[vpx.TensorTree, Callable[[vpx.TensorTree], vpx.TensorTree]]:
        events.append("linearize")
        primal = tensor_function(params)

        def jvp_function(vector: vpx.TensorTree) -> vpx.TensorTree:
            events.append("jvp")

            return {"y": tree_leaves(vector)[0] * 7.0}

        return primal, jvp_function

    monkeypatch.setattr(runtime_module.torch.func, "linearize", fake_linearize)
    factory = vpx.standard_operation_factory(
        ops.jvp("jvp", "function", aggregation="none"),
        params={"w": torch.tensor([3.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"function": function},
    )
    operation = factory(
        vpx.Candidate(
            "jvp",
            candidate_id,
            {
                "jvp.path": "torch_func_linearize",
                "jvp.linearize_reuse": reuse,
                **torch_func_settings(requires_forward_ad=True),
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([2.0], dtype=torch.float64)},
    )

    return events, operation(), operation()


@pytest.mark.parametrize(
    ("reuse", "candidate_id", "expected_events"),
    [
        (
            "reuse_at_same_primal",
            "reuse-linearize",
            ["linearize", "function", "jvp", "jvp"],
        ),
        (
            "none",
            "rebuild-linearize",
            ["linearize", "function", "jvp", "linearize", "function", "jvp"],
        ),
    ],
)
def test_jvp_linearize_reuse_controls_linear_function_builds(
    monkeypatch: pytest.MonkeyPatch,
    reuse: str,
    candidate_id: str,
    expected_events: list[str],
) -> None:
    events, first, second = jvp_linearize_reuse_result(
        monkeypatch,
        reuse,
        candidate_id,
    )

    assert events == expected_events
    assert torch.allclose(
        tree_leaves(first)[0],
        torch.tensor([14.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(second)[0],
        torch.tensor([14.0], dtype=torch.float64),
    )


def test_vjp_closure_reuse_axis_requires_torch_func_vjp_path() -> None:
    registry = vpx.standard_axis_registry()
    admitted = registry.admit(
        vpx.Candidate(
            "vjp",
            "good",
            {
                "vjp.path": "torch_func_vjp",
                "vjp.closure_reuse": "reuse_vjp_closure_at_same_primal",
                **torch_func_settings(requires_forward_ad=False),
            },
        )
    )
    rejected = registry.admit(
        vpx.Candidate(
            "vjp",
            "bad",
            {
                "vjp.path": "autograd_grad_outputs",
                "vjp.closure_reuse": "reuse_vjp_closure_at_same_primal",
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"
    assert rejected.admission_error == (
        "reuse_vjp_closure_at_same_primal requires torch_func_vjp"
    )


def vjp_closure_reuse_result(
    monkeypatch: pytest.MonkeyPatch,
    reuse: str,
    candidate_id: str,
) -> tuple[list[str], vpx.TensorTree, vpx.TensorTree]:
    events = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert buffers == {}
        assert batch["scale"] == pytest.approx(1.0)
        assert context.family == "vjp"
        events.append("function")

        return {"y": params["w"] * 2.0}

    def fake_vjp(
        tensor_function: Callable[[vpx.ParameterTree], vpx.TensorTree],
        params: vpx.ParameterTree,
        *,
        has_aux: bool,
    ) -> tuple[vpx.TensorTree, Callable[[vpx.TensorTree], tuple[vpx.TensorTree]]]:
        assert has_aux is False
        events.append("vjp")
        output = tensor_function(params)

        def pullback(cotangent: vpx.TensorTree) -> tuple[vpx.TensorTree]:
            events.append("pullback")

            return ({"w": tree_leaves(cotangent)[0] * 11.0},)

        return output, pullback

    monkeypatch.setattr(runtime_module.torch.func, "vjp", fake_vjp)
    factory = vpx.standard_operation_factory(
        ops.vjp("vjp", "function", aggregation="none"),
        params={"w": torch.tensor([3.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"function": function},
    )
    operation = factory(
        vpx.Candidate(
            "vjp",
            candidate_id,
            {
                "vjp.path": "torch_func_vjp",
                "vjp.closure_reuse": reuse,
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"y": torch.tensor([2.0], dtype=torch.float64)},
    )

    return events, operation(), operation()


@pytest.mark.parametrize(
    ("reuse", "candidate_id", "expected_events"),
    [
        (
            "reuse_vjp_closure_at_same_primal",
            "reuse-pullback",
            ["vjp", "function", "pullback", "pullback"],
        ),
        (
            "none",
            "rebuild-pullback",
            ["vjp", "function", "pullback", "vjp", "function", "pullback"],
        ),
    ],
)
def test_vjp_closure_reuse_controls_pullback_builds(
    monkeypatch: pytest.MonkeyPatch,
    reuse: str,
    candidate_id: str,
    expected_events: list[str],
) -> None:
    events, first, second = vjp_closure_reuse_result(
        monkeypatch,
        reuse,
        candidate_id,
    )

    assert events == expected_events
    assert torch.allclose(
        tree_leaves(first)[0],
        torch.tensor([22.0], dtype=torch.float64),
    )
    assert torch.allclose(
        tree_leaves(second)[0],
        torch.tensor([22.0], dtype=torch.float64),
    )


def _run_function_vectorization_case(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    function_objectives: Mapping[str, vpx.FunctionObjective],
) -> vpx.TensorTree:
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
        buffers={},
        function_objectives=function_objectives,
    )

    return factory(
        vpx.Candidate(
            operator.family,
            "vectorized",
            settings,
            admission_status="passed",
        ),
        batch,
        vector,
    )()


THREE_VECTOR_ROWS = ((1.0, 0.0), (0.0, 2.0), (-1.0, 3.0))
FIVE_VECTOR_ROWS = (*THREE_VECTOR_ROWS, (4.0, -2.0), (0.5, 0.25))
COMPOSITION_THREE_VECTOR_ROWS = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
COMPOSITION_FIVE_VECTOR_ROWS = (
    *COMPOSITION_THREE_VECTOR_ROWS,
    (7.0, 8.0),
    (9.0, 10.0),
)
SCORE_GRADIENT_ROWS = ((1.0, 0.0), (2.0, -1.0), (0.5, 3.0))
FISHER_ZERO_SCORE_ROWS = ((1.0, 2.0), (3.0, 5.0))
DENSE_2X2_ROWS = ((4.0, 1.0), (1.0, 3.0))
INNER_LEFT_BLOCK_ROWS = ((1.0, 2.0), (0.5, -1.0))
INNER_RIGHT_BLOCK_ROWS = ((3.0, 2.0, -0.5), (-1.0, 1.0, 0.75))
BLOCK_MANUAL_VECTOR_SETTINGS = {
    "vectorization.mode": "manual_batch",
    "vectorization.batch_size": 2,
    "vectorization.in_dims": ({"w": 0}, {"w": 1}),
}
BLOCK_VMAP_VECTOR_SETTINGS = {
    "vectorization.mode": "vmap",
    "vectorization.vmap_chunk_size": 2,
    "vectorization.in_dims": ({"w": 0}, {"w": 1}),
    **torch_func_settings(requires_forward_ad=False),
}
COLUMN_BLOCKS = itemgetter((slice(None), slice(1)), (slice(None), slice(1, None)))


def _jvp_vjp_expected(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    return tuple((4.0 * x, -2.0 * y) for x, y in rows)


def _ggn_expected(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((16.0 * x, 4.0 * y) for x, y in rows)


def _float_rows(rows: Sequence[Sequence[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def _vector_rows(key: str, rows: Sequence[Sequence[float]]) -> dict[str, torch.Tensor]:
    return {key: _float_rows(rows)}


def _zero_w_params(width: int) -> dict[str, torch.Tensor]:
    return {"w": torch.zeros(width, dtype=torch.float64)}


@pytest.mark.parametrize(
    (
        "operator",
        "settings",
        "batch",
        "vector",
        "function_objectives",
        "expected",
    ),
    [
        (
            ops.jvp("jvp", "function", aggregation="none"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            {"scale": 1.0},
            _vector_rows("w", THREE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.vjp("vjp", "function", aggregation="none"),
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"y": 0},
            },
            {"scale": 1.0},
            _vector_rows("y", THREE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            {
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            {"loss_hessian": torch.eye(2, dtype=torch.float64)},
            _vector_rows("w", THREE_VECTOR_ROWS),
            {"model_output": square_tensor_function},
            _float_rows(_ggn_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.jvp("jvp", "function", aggregation="none"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                **vmap_settings({"w": 0}, chunk_size=2),
            },
            {"scale": 1.0},
            _vector_rows("w", THREE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.vjp("vjp", "function", aggregation="none"),
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                "vectorization.in_dims": {"y": 0},
                "vectorization.mode": "vmap",
                "vectorization.randomness": "error",
                "vectorization.vmap_chunk_size": 2,
            },
            {"scale": 1.0},
            _vector_rows("y", THREE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            {
                **ggn_dense_kernel_settings(),
                "vectorization.in_dims": {"w": 0},
                "vectorization.mode": "vmap",
                "vectorization.randomness": "error",
                "vectorization.vmap_chunk_size": 2,
            },
            {"loss_hessian": torch.eye(2, dtype=torch.float64)},
            _vector_rows("w", THREE_VECTOR_ROWS),
            {"model_output": square_tensor_function},
            _float_rows(_ggn_expected(THREE_VECTOR_ROWS)),
        ),
        (
            ops.jvp("jvp", "function", aggregation="none"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            {"scale": 1.0},
            _vector_rows("w", FIVE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(FIVE_VECTOR_ROWS)),
        ),
        (
            ops.vjp("vjp", "function", aggregation="none"),
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"y": 0},
            },
            {"scale": 1.0},
            _vector_rows("y", FIVE_VECTOR_ROWS),
            {"function": square_function},
            _float_rows(_jvp_vjp_expected(FIVE_VECTOR_ROWS)),
        ),
        (
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            {
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            {"loss_hessian": torch.eye(2, dtype=torch.float64)},
            _vector_rows("w", FIVE_VECTOR_ROWS),
            {"model_output": square_tensor_function},
            _float_rows(_ggn_expected(FIVE_VECTOR_ROWS)),
        ),
    ],
)
def test_jvp_vjp_ggn_vectorization_runs_batched_vectors(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    function_objectives: Mapping[str, vpx.FunctionObjective],
    expected: torch.Tensor,
) -> None:
    result = _run_function_vectorization_case(
        operator,
        settings,
        batch,
        vector,
        function_objectives,
    )

    torch.testing.assert_close(tree_leaves(result)[0], expected)


@pytest.mark.parametrize(
    ("operator", "settings", "vector", "message"),
    [
        (
            ops.jvp("jvp", "function", aggregation="none"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "vectorization.mode": "manual_batch",
                "vectorization.in_dims": {"w": 0},
            },
            {"w": torch.tensor([[3.0]], dtype=torch.float64)},
            "vectorization.batch_size",
        ),
        (
            ops.vjp("vjp", "function", aggregation="none"),
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 0,
                "vectorization.in_dims": {"y": 0},
            },
            {"y": torch.tensor([[3.0]], dtype=torch.float64)},
            "positive",
        ),
        (
            ops.jvp("jvp", "function", aggregation="none"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "vectorization.mode": "single_loop",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            {"w": torch.tensor([[3.0]], dtype=torch.float64)},
            "manual_batch",
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "manual_batch",
                "vectorization.in_dims": {"w": 0},
            },
            {"w": torch.tensor([[3.0]], dtype=torch.float64)},
            "vectorization.batch_size",
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 0,
                "vectorization.in_dims": {"w": 0},
            },
            {"w": torch.tensor([[3.0]], dtype=torch.float64)},
            "positive",
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "single_loop",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            {"w": torch.tensor([[3.0]], dtype=torch.float64)},
            "manual_batch",
        ),
    ],
)
def test_vectorization_manual_batch_settings_are_validated(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    vector: vpx.TensorTree,
    message: str,
) -> None:
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        function_objectives={"function": square_function},
    )

    with pytest.raises(vp.MaterializationError, match=message):
        factory(
            vpx.Candidate(
                operator.family,
                "bad-manual-batch",
                settings,
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_ggn_vectorization_rejects_inner_compile_boundary() -> None:
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"model_output": square_tensor_function},
    )

    with pytest.raises(vp.MaterializationError, match="ggn_jvp"):
        factory(
            vpx.Candidate(
                "ggn",
                "bad-inner-compile",
                {
                    **ggn_dense_kernel_settings(),
                    "vectorization.mode": "single_loop",
                    "vectorization.in_dims": {"w": 0},
                    **compile_settings(boundary="ggn_jvp"),
                },
                admission_status="passed",
            ),
            {"loss_hessian": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([[1.0]], dtype=torch.float64)},
        )


@pytest.mark.parametrize(
    ("operator", "settings", "batch"),
    [
        (
            score_terms_fisher_sum("fisher", "scores"),
            fisher_settings("materialize_score_gradients"),
            {"score_gradients": torch.eye(2, dtype=torch.float64)},
        ),
        (
            score_terms_sampled_fisher_sum(
                "sampled",
                "scores",
                sample_source="fixed_sample_table",
            ),
            sampled_fisher_settings(
                "materialize_score_gradients",
                sample_source="fixed_sample_table",
            ),
            {"sampled_score_gradients": torch.eye(2, dtype=torch.float64)},
        ),
        (
            empirical_fisher_sum("empirical", "scores"),
            empirical_dense_settings(),
            {"per_example_gradients": torch.eye(2, dtype=torch.float64)},
        ),
    ],
)
@pytest.mark.parametrize(
    ("candidate_id", "mode_settings"),
    [
        (
            "single-loop-vectors",
            {
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
        ),
        (
            "manual-batch-vectors",
            {
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
        ),
        (
            "vmap-vectors",
            {
                **vmap_settings({"w": 0}, chunk_size=2),
                **torch_func_settings(requires_forward_ad=False),
            },
        ),
    ],
)
def test_fisher_family_vectorization_runs_batched_vectors(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    candidate_id: str,
    mode_settings: Mapping[str, object],
) -> None:
    vector = {
        "w": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float64,
        )
    }
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.zeros(2, dtype=torch.float64)},
        buffers={},
    )
    result = factory(
        vpx.Candidate(
            operator.family,
            candidate_id,
            {
                **settings,
                **mode_settings,
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"])


def test_parameter_order_vector_is_prepared_before_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = runtime_module._build_parameter_order_vector

    def counted_build(execution: runtime_module.StandardExecution) -> torch.Tensor:
        calls.append(execution.candidate.candidate_id)

        return original(execution)

    monkeypatch.setattr(
        runtime_module,
        "_build_parameter_order_vector",
        counted_build,
    )
    operator = score_terms_fisher_sum("fisher", "scores")
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.zeros(2, dtype=torch.float64)},
        buffers={},
    )
    operation = factory(
        vpx.Candidate(
            operator.family,
            "cached-vector",
            fisher_settings("materialize_score_gradients"),
            admission_status="passed",
        ),
        {"score_gradients": torch.eye(2, dtype=torch.float64)},
        {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
    )

    assert calls == ["cached-vector"]

    first = operation()
    second = operation()

    expected = torch.tensor([1.0, 2.0], dtype=torch.float64)

    torch.testing.assert_close(tree_leaves(first)[0], expected)
    torch.testing.assert_close(tree_leaves(second)[0], expected)
    assert calls == ["cached-vector"]


def test_ggn_vjp_path_axis_requires_valid_jvp_hessian_vjp_row() -> None:
    registry = vpx.standard_axis_registry()
    admitted = registry.admit(
        vpx.Candidate(
            "ggn",
            "good",
            {
                **ggn_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
            },
        )
    )
    rejected_wrong_key = registry.admit(
        vpx.Candidate(
            "ggn",
            "dense-on-jvp-key",
            {
                "ggn.jvp_path": "dense_global",
                "ggn.vjp_path": "autograd_grad_outputs",
            },
        )
    )
    rejected_missing_fields = registry.admit(
        vpx.Candidate(
            "ggn",
            "missing-torch-func-fields",
            {
                "ggn.jvp_path": "torch_func_jvp",
                "ggn.vjp_path": "torch_func_vjp",
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected_wrong_key.admission_status == "failed"
    assert rejected_wrong_key.admission_error == (
        "candidate axis value is not allowed: ggn.jvp_path"
    )
    assert rejected_missing_fields.admission_status == "failed"
    assert rejected_missing_fields.admission_error is not None
    assert (
        "torch.func admission fields missing" in rejected_missing_fields.admission_error
    )


def test_ggn_vjp_path_autograd_grad_outputs_matches_torch_func_vjp() -> None:
    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0].pow(2), 3.0 * params["w"][0]))

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"model_output": function},
    )
    batch = {"loss_hessian": torch.diag(torch.tensor([5.0, 7.0], dtype=torch.float64))}
    vector = {"w": torch.tensor([11.0], dtype=torch.float64)}
    torch_func_result = factory(
        vpx.Candidate(
            "ggn",
            "torch-func-vjp",
            {
                **ggn_settings("torch_func_jvp", vjp_path="torch_func_vjp"),
                **torch_func_settings(requires_forward_ad=True),
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    autograd_result = factory(
        vpx.Candidate(
            "ggn",
            "autograd-vjp",
            {
                **ggn_settings("torch_func_jvp", vjp_path="autograd_grad_outputs"),
                **torch_func_settings(requires_forward_ad=True),
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    assert torch.allclose(
        tree_leaves(torch_func_result)[0], tree_leaves(autograd_result)[0]
    )
    assert torch.allclose(
        tree_leaves(autograd_result)[0],
        torch.tensor([1573.0], dtype=torch.float64),
    )


def test_ggn_vjp_path_is_required_for_jvp_hessian_vjp_rows() -> None:
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"model_output": square_function},
    )

    with pytest.raises(vp.MaterializationError, match=r"ggn\.vjp_path is required"):
        factory(
            vpx.Candidate(
                "ggn",
                "missing-vjp-path",
                {
                    "ggn.jvp_path": "torch_func_jvp",
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {"loss_hessian": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )()


def test_standard_operation_factory_enforces_direct_admission_fields() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {}
    batch = {"scale": 1.0}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    jvp_factory = standard_factory(
        params,
        buffers,
        ops.jvp("jvp", "function", aggregation="none"),
        function_objectives={"function": square_function},
    )
    gradient_factory = standard_factory(
        params,
        buffers,
        ops.gradient("gradient", "loss", aggregation="sum"),
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match=r"jvp\.path"):
        jvp_factory(
            vpx.Candidate(
                "jvp",
                "missing-path",
                {},
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="missing fields"):
        jvp_factory(
            vpx.Candidate(
                "jvp",
                "missing-fields",
                jvp_settings("torch_func_jvp"),
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="contains_autograd_call"):
        jvp_factory(
            vpx.Candidate(
                "jvp",
                "invalid-torch-func",
                {
                    **jvp_settings("torch_func_jvp"),
                    **torch_func_settings(requires_forward_ad=True),
                    "contains_autograd_call": True,
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="unsupported"):
        jvp_factory(
            vpx.Candidate(
                "jvp",
                "unsupported-forward-ad",
                {
                    **jvp_settings("forward_ad_dual"),
                    "requires_forward_ad": True,
                    "forward_ad_supported": False,
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="missing fields"):
        gradient_factory(
            vpx.Candidate(
                "gradient",
                "functional-call-field",
                {**gradient_settings(), "tie_weights": True},
                admission_status="passed",
            ),
            batch,
            vector,
        )()


def test_standard_operation_factory_rejects_singleton_operator_path() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {}
    batch = {"scale": 1.0}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    cotangent = {"y": torch.tensor([4.0], dtype=torch.float64)}
    metric_batch = {"metric_matrix": torch.eye(1, dtype=torch.float64)}
    cases = (
        (
            ops.gradient("gradient", "loss", aggregation="sum"),
            {"scalar_objectives": {"loss": quadratic_scalar}},
            batch,
            vector,
            "autograd_grad",
        ),
        (
            ops.vjp("vjp", "function", aggregation="none"),
            {"function_objectives": {"function": square_function}},
            batch,
            cotangent,
            "torch_func_vjp",
        ),
        (
            ops.metric(
                "metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
            ),
            {},
            metric_batch,
            vector,
            "dense_metric",
        ),
        (
            ops.inverse_metric(
                "inverse_metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
                damping=0.0,
            ),
            {},
            metric_batch,
            vector,
            "dense_inverse_metric",
        ),
    )

    for operator, kwargs, runtime_batch, runtime_vector, path in cases:
        factory = standard_factory(params, buffers, operator, **kwargs)

        with pytest.raises(vp.MaterializationError, match="operator_path"):
            factory(
                vpx.Candidate(
                    operator.family,
                    "path-supplied",
                    {"operator_path": path},
                    admission_status="passed",
                ),
                runtime_batch,
                runtime_vector,
            )()


def test_standard_operation_factory_requires_metric_spec_path_keys() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {"metric_matrix": torch.eye(1, dtype=torch.float64)}
    cases = (
        (
            ops.metric(
                "metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
            ),
            "metric.multiply_path",
        ),
        (
            ops.inverse_metric(
                "inverse_metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
                damping=0.0,
            ),
            "inverse_metric.solve_path",
        ),
    )

    for operator, required_key in cases:
        factory = standard_factory(params, {}, operator)

        with pytest.raises(vp.MaterializationError, match=required_key):
            factory(
                vpx.Candidate(
                    operator.family,
                    "missing-path",
                    {},
                    admission_status="passed",
                ),
                batch,
                vector,
            )()


def test_standard_reference_check_requires_thresholds() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    with pytest.raises(vp.MaterializationError):
        vpx.standard_reference_check(
            ops.hvp("hvp", "loss", aggregation="sum"),
            params=params,
            buffers={},
            thresholds={},
            scalar_objectives={"loss": quadratic_scalar},
        )


def test_standard_reference_check_preserves_candidate_context_identity() -> None:
    calls = []
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"]
        calls.append(context.candidate_id)

        return params["w"].pow(2).sum()

    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
            "symmetry_max_abs_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar},
    )
    candidate = vpx.Candidate(
        "hvp",
        "candidate-row",
        {
            **hvp_settings("jvp_grad"),
            **torch_func_settings(requires_forward_ad=True),
        },
        admission_status="passed",
    )

    result = check(
        candidate,
        {
            "scale": 1.0,
            "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert calls
    assert all(call == "candidate-row" for call in calls)
    assert "directional_abs_diff" in result.measurements
    assert "directional_rel_diff" in result.measurements


def test_hvp_reference_check_records_symmetry_error() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check_without_symmetry_threshold = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
        vpx.Candidate(
            "hvp",
            "row",
            {
                **hvp_settings("jvp_grad"),
                **torch_func_settings(requires_forward_ad=True),
            },
            admission_status="passed",
        ),
        {
            "scale": 1.0,
            "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["symmetry_max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError, match="symmetry_max_abs_diff"):
        check_without_symmetry_threshold(
            vpx.Candidate(
                "hvp",
                "row",
                {
                    **hvp_settings("jvp_grad"),
                    **torch_func_settings(requires_forward_ad=True),
                },
                admission_status="passed",
            ),
            {
                "scale": 1.0,
                "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="symmetry_vector"):
        check(
            vpx.Candidate(
                "hvp",
                "row",
                {
                    **hvp_settings("jvp_grad"),
                    **torch_func_settings(requires_forward_ad=True),
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_hvp_reference_check_accepts_functional_hvp_path() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
        vpx.Candidate(
            "hvp",
            "row",
            hvp_settings("autograd_functional_hvp"),
            admission_status="passed",
        ),
        {
            "scale": 1.0,
            "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["directional_abs_diff"] < 1e-3
    assert result.measurements["symmetry_max_abs_diff"] == pytest.approx(0.0)


def test_hvp_runtime_accepts_spec_path_keys_and_rejects_mixed_path_keys() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {
        "scale": 1.0,
        "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
    }
    factory = quadratic_hvp_factory(params)
    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
    reverse = passed_candidate(
        "hvp",
        "reverse",
        {"hvp.path": "reverse_over_reverse"},
    )
    functional = passed_candidate(
        "hvp",
        "functional",
        {"hvp.path": "autograd_functional_hvp"},
    )
    vhp = passed_candidate(
        "hvp",
        "vhp",
        {"hvp.path": "autograd_functional_vhp"},
    )
    jvp_grad = passed_candidate(
        "hvp",
        "jvp-grad",
        {
            "hvp.path": "jvp_grad",
            **torch_func_settings(requires_forward_ad=True),
        },
    )
    mixed = passed_candidate(
        "hvp",
        "mixed",
        {"hvp.path": "reverse_over_reverse", "operator_path": "functional_hvp"},
    )

    torch.testing.assert_close(
        tree_leaves(factory(reverse, batch, vector)())[0],
        torch.tensor([6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tree_leaves(factory(functional, batch, vector)())[0],
        torch.tensor([6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tree_leaves(factory(vhp, batch, vector)())[0],
        torch.tensor([6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tree_leaves(factory(jvp_grad, batch, vector)())[0],
        torch.tensor([6.0], dtype=torch.float64),
    )
    assert check(jvp_grad, batch, vector).measurements["max_abs_diff"] == pytest.approx(
        0.0
    )

    with pytest.raises(vp.MaterializationError, match="operator_path"):
        factory(mixed, batch, vector)()


def test_hvp_jvp_grad_does_not_call_functional_hvp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}

    def forbidden_functional_hvp(*_: Any, **__: Any) -> Any:
        message = "jvp_grad HVP called functional HVP"
        raise AssertionError(message)

    monkeypatch.setattr(
        runtime_module.torch.autograd.functional,
        "hvp",
        forbidden_functional_hvp,
    )
    monkeypatch.setattr(
        runtime_module.torch.autograd.functional,
        "vhp",
        forbidden_functional_hvp,
    )
    result = run_quadratic_hvp(
        "jvp-grad",
        {
            **hvp_settings("jvp_grad"),
            **torch_func_settings(requires_forward_ad=True),
        },
        params,
        {"scale": 1.0},
        vector,
    )

    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([6.0], dtype=torch.float64),
    )


def test_hvp_linearize_grad_reuses_prepared_gradient_closure() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {"scale": 1.0}
    result = run_quadratic_hvp(
        "reuse-gradient-closure",
        hvp_gradient_reuse_settings(),
        params,
        batch,
        vector,
    )

    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([6.0], dtype=torch.float64),
    )


def _run_hvp_vectorization_case(
    settings: Mapping[str, object],
    vector: dict[str, torch.Tensor],
) -> vpx.TensorTree:
    return run_quadratic_hvp(
        "vectorized",
        settings,
        {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
        {"scale": 3.0},
        vector,
    )


@pytest.mark.parametrize(
    ("settings", "vector"),
    [
        (
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            _vector_rows("w", THREE_VECTOR_ROWS),
        ),
        (
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            _vector_rows("w", FIVE_VECTOR_ROWS),
        ),
        (
            {
                **hvp_gradient_reuse_settings(),
                **vmap_settings({"w": 0}, chunk_size=1),
            },
            _vector_rows("w", THREE_VECTOR_ROWS),
        ),
    ],
)
def test_hvp_vectorization_runs_batched_vectors(
    settings: Mapping[str, object],
    vector: dict[str, torch.Tensor],
) -> None:
    result = _run_hvp_vectorization_case(settings, vector)

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"] * 6.0)


@pytest.mark.parametrize(
    ("case_id", "reuse_settings"),
    [
        (
            "retain-graph-vectors",
            {
                "hvp.graph_schedule": "retain_graph_across_vectors",
                "hvp.primal_reuse": "reuse_primal",
            },
        ),
        (
            "reuse-primal-vectors",
            {
                "hvp.graph_schedule": "rebuild_graph_per_vector",
                "hvp.primal_reuse": "reuse_primal",
            },
        ),
    ],
)
def test_hvp_reverse_reuse_vector_policies(
    case_id: str,
    reuse_settings: dict[str, object],
) -> None:
    params = {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)}
    vector = _vector_rows("w", THREE_VECTOR_ROWS)
    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )
    result = factory(
        vpx.Candidate(
            "hvp",
            case_id,
            {
                **hvp_settings("reverse_over_reverse"),
                **reuse_settings,
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            admission_status="passed",
        ),
        {"scale": 3.0},
        vector,
    )()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"] * 6.0)


def test_hvp_vmap_vectorization_rejects_non_linearized_path() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([[3.0]], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(vp.MaterializationError, match="linearize_grad HVP"):
        factory(
            passed_candidate(
                "hvp",
                "bad-vmap-vectors",
                {
                    **hvp_settings("reverse_over_reverse"),
                    **torch_func_settings(requires_forward_ad=False),
                    "vectorization.mode": "vmap",
                    "vectorization.vmap_chunk_size": 1,
                    "vectorization.in_dims": {"w": 0},
                },
            ),
            {"scale": 1.0},
            vector,
        )()


def test_hvp_retain_graph_requires_primal_reuse() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([[3.0]], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(vp.MaterializationError, match=r"hvp[.]primal_reuse"):
        factory(
            passed_candidate(
                "hvp",
                "retain-with-recompute",
                {
                    **hvp_settings("reverse_over_reverse"),
                    "hvp.graph_schedule": "retain_graph_across_vectors",
                    "hvp.primal_reuse": "recompute_primal",
                    "vectorization.mode": "single_loop",
                    "vectorization.in_dims": {"w": 0},
                },
            ),
            {"scale": 1.0},
            vector,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {
            **hvp_settings("jvp_grad"),
            **torch_func_settings(requires_forward_ad=True),
            "hvp.gradient_reuse": "reuse_gradient_closure",
        },
        {
            **hvp_settings("jvp_grad"),
            **torch_func_settings(requires_forward_ad=True),
            "hvp.graph_schedule": "retain_graph_across_vectors",
        },
        {
            **hvp_settings("jvp_grad"),
            **torch_func_settings(requires_forward_ad=True),
            "hvp.primal_reuse": "reuse_primal",
        },
    ],
)
def test_hvp_reuse_rejects_rows_without_required_execution_surface(
    settings: dict[str, object],
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(
        vp.MaterializationError,
        match=r"linearize_grad|reverse_over_reverse",
    ):
        factory(
            passed_candidate(
                "hvp",
                "bad-reuse",
                settings,
            ),
            {"scale": 1.0},
            vector,
        )


def test_gradient_reference_check_records_directional_agreement() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
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
        vpx.Candidate(
            "gradient",
            "row",
            {"gradient.path": "torch_autograd_grad"},
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["directional_abs_diff"] < 1e-3


def test_jvp_reference_check_records_finite_difference_agreement() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.jvp("jvp", "function", aggregation="none"),
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
        vpx.Candidate(
            "jvp",
            "row",
            {
                "jvp.path": "forward_ad_dual",
                "requires_forward_ad": True,
                "forward_ad_supported": True,
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["directional_abs_diff"] < 1e-3


def test_vjp_reference_check_records_dot_identity() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check_without_inner_threshold = vpx.standard_reference_check(
        ops.vjp("vjp", "function", aggregation="none"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
        },
        function_objectives={"function": square_function},
    )
    check = vpx.standard_reference_check(
        ops.vjp("vjp", "function", aggregation="none"),
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
        vpx.Candidate(
            "vjp",
            "row",
            {
                "vjp.path": "torch_func_vjp",
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        {
            "scale": 1.0,
            "tangent_vector": {"w": torch.tensor([3.0], dtype=torch.float64)},
        },
        {"y": torch.tensor([4.0], dtype=torch.float64)},
    )

    assert result.measurements["inner_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError, match="inner_abs_diff"):
        check_without_inner_threshold(
            vpx.Candidate(
                "vjp",
                "row",
                {
                    "vjp.path": "torch_func_vjp",
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {
                "scale": 1.0,
                "tangent_vector": {"w": torch.tensor([3.0], dtype=torch.float64)},
            },
            {"y": torch.tensor([4.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="tangent_vector"):
        check(
            vpx.Candidate(
                "vjp",
                "row",
                {
                    "vjp.path": "torch_func_vjp",
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"y": torch.tensor([4.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check(
            vpx.Candidate(
                "vjp",
                "row",
                {
                    "vjp.path": "torch_func_vjp",
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {
                "scale": 1.0,
                "tangent_vector": {"w": torch.tensor([0.0], dtype=torch.float64)},
            },
            {"y": torch.tensor([4.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check(
            vpx.Candidate(
                "vjp",
                "row",
                {
                    "vjp.path": "torch_func_vjp",
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            {
                "scale": 1.0,
                "tangent_vector": {"w": torch.tensor([3.0], dtype=torch.float64)},
            },
            {"y": torch.tensor([0.0], dtype=torch.float64)},
        )


def test_vhp_reference_check_requires_symmetry_and_directional_checks() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    check_without_symmetry = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
    check_with_symmetry = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
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
    candidate = vpx.Candidate(
        "hvp",
        "vhp",
        hvp_settings("autograd_functional_vhp"),
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

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check_with_symmetry(
            candidate,
            {
                "scale": 1.0,
                "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([0.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check_with_symmetry(
            candidate,
            {
                "scale": 1.0,
                "symmetry_vector": {"w": torch.tensor([0.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_standard_reference_check_honors_strict_thresholds() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"]
        multiplier = 1.000005 if context.settings["hvp.path"] == "jvp_grad" else 1.0

        return multiplier * params["w"].pow(2).sum()

    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-9,
            "max_rel_diff": 1e-9,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
            "symmetry_max_abs_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar},
    )
    candidate = vpx.Candidate(
        "hvp",
        "candidate-row",
        {
            "hvp.path": "jvp_grad",
            **torch_func_settings(requires_forward_ad=True),
        },
        admission_status="passed",
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            candidate,
            {
                "scale": 1.0,
                "symmetry_vector": {"w": torch.tensor([4.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_reference_check_applies_numeric_error_bound_fields() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    operator = ops.gradient("gradient", "loss", aggregation="sum")
    candidate = vpx.Candidate(
        "gradient",
        "high-matmul",
        {
            **gradient_settings(),
            "numeric.float32_matmul_precision": "high",
        },
        admission_status="passed",
    )
    thresholds = {
        "max_abs_diff": 1e-4,
        "max_rel_diff": 1e-3,
        "directional_abs_diff": 1e-3,
        "directional_rel_diff": 1e-2,
    }
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds=thresholds,
        numeric_bound_fields=numeric_bound_fields(),
        scalar_objectives={"loss": quadratic_scalar},
    )
    result = check(
        candidate,
        {"scale": torch.tensor(2.0, dtype=torch.float64)},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["numeric_error_bound_abs"] == pytest.approx(
        1e-8 / (1.0 - 1e-8)
    )
    assert result.thresholds == thresholds

    loose_bound_check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds=thresholds,
        numeric_bound_fields=numeric_bound_fields(epsilon=0.5),
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.ReferenceFailedError, match="bound exceeds threshold"):
        loose_bound_check(
            candidate,
            {"scale": torch.tensor(2.0, dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_dtype_accumulation_requires_numeric_error_bound_fields() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    operator = ops.gradient("gradient", "loss", aggregation="sum")
    candidate = vpx.Candidate(
        "gradient",
        "bf16-accumulation",
        {
            **gradient_settings(),
            "dtype.accumulation": "bf16",
        },
        admission_status="passed",
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-4,
            "max_rel_diff": 1e-3,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-2,
        },
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.ReferenceFailedError, match="fields are missing"):
        check(
            candidate,
            {"scale": torch.tensor(2.0, dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_standard_reference_check_low_precision_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "metric",
                "float32",
                {**metric_settings(), "dtype.output": "fp32"},
                admission_status="passed",
            ),
            {"metric_matrix": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.00000006], dtype=torch.float64)},
        )


def test_metric_reference_check_rejects_nonsymmetric_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "metric",
                "row",
                metric_settings(),
                admission_status="passed",
            ),
            {
                "metric_matrix": torch.tensor(
                    [[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64
                )
            },
            {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        )


def test_metric_reference_check_rejects_indefinite_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
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
            vpx.Candidate(
                "metric",
                "row",
                metric_settings(),
                admission_status="passed",
            ),
            {
                "metric_matrix": torch.diag(
                    torch.tensor([1.0, -0.1], dtype=torch.float64)
                )
            },
            {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)},
        )


def test_standard_operation_factory_rejects_missing_declared_batch_input() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert params["w"] is not None
        assert buffers == {}
        assert batch == {}
        assert context.family == "ggn"
        calls.append("called")

        return params["w"]

    ggn_factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.MaterializationError, match="loss_hessian"):
        ggn_factory(
            vpx.Candidate(
                "ggn",
                "row",
                ggn_dense_kernel_settings(),
                admission_status="passed",
            ),
            {},
            vector,
        )

    assert calls == []


def test_dense_standard_paths_reject_nonfinite_inputs() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return params["w"]

    ggn_factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    fisher_factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    metric_factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="nonfinite"):
        ggn_factory(
            vpx.Candidate(
                "ggn",
                "row",
                ggn_dense_kernel_settings(),
                admission_status="passed",
            ),
            {"loss_hessian": torch.tensor([[torch.nan]], dtype=torch.float64)},
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="nonfinite"):
        fisher_factory(
            vpx.Candidate(
                "fisher",
                "row",
                fisher_settings("materialize_score_gradients"),
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
            vpx.Candidate(
                "metric",
                "row",
                metric_settings(),
                admission_status="passed",
            ),
            {"metric_matrix": torch.tensor([[torch.nan]], dtype=torch.float64)},
            vector,
        )()


@pytest.mark.parametrize(
    "kernel",
    ["dense_global", "streaming_global", "two_pass_chunked_global"],
)
def test_ggnvp_closed_form_ce_kl_kernels_match_dense_loss_hessian(
    kernel: str,
) -> None:
    params = {
        "logits": torch.tensor(
            [[0.5, -1.0, 2.0], [1.25, -0.75, 0.0]],
            dtype=torch.float64,
        )
    }
    vector = {
        "logits": torch.tensor(
            [[0.25, -0.5, 1.0], [-1.0, 0.75, 0.5]],
            dtype=torch.float64,
        )
    }

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "ggn"

        return params["logits"]

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            f"closed-form-{kernel}",
            ggn_closed_form_ce_kl_settings(kernel),
            admission_status="passed",
        ),
        {},
        vector,
    )()
    expected = dense_softmax_ce_kl_loss_hessian(params["logits"]) @ vector[
        "logits"
    ].reshape(-1)
    result_map = tensor_mapping(result)

    torch.testing.assert_close(
        result_map["logits"],
        expected.reshape_as(params["logits"]),
    )


@pytest.mark.parametrize(
    "kernel",
    ["dense_global", "streaming_global", "two_pass_chunked_global"],
)
def test_ggnvp_closed_form_ce_kl_token_blocks_match_dense_loss_hessian(
    kernel: str,
) -> None:
    params = {
        "logits": torch.tensor(
            [
                [[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]],
                [[1.25, -0.75, 0.0], [-1.0, 2.0, 0.25]],
            ],
            dtype=torch.float64,
        )
    }
    vector = {
        "logits": torch.tensor(
            [
                [[0.25, -0.5, 1.0], [0.75, -1.25, 0.5]],
                [[-1.0, 0.75, 0.5], [0.5, -0.25, -0.75]],
            ],
            dtype=torch.float64,
        )
    }

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "ggn"

        return params["logits"]

    settings = {
        **ggn_closed_form_ce_kl_settings(kernel),
        "schedule.per_token": "loop",
        "chunk.token_block_size": 2,
    }
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            f"token-block-{kernel}",
            settings,
            admission_status="passed",
        ),
        {},
        vector,
    )()
    expected = dense_softmax_ce_kl_loss_hessian(params["logits"]) @ vector[
        "logits"
    ].reshape(-1)
    result_map = tensor_mapping(result)

    torch.testing.assert_close(
        result_map["logits"],
        expected.reshape_as(params["logits"]),
    )


@pytest.mark.parametrize(
    "kernel",
    ["dense_global", "streaming_global", "two_pass_chunked_global"],
)
def test_ggnvp_closed_form_ce_kl_reference_matches_dense_anchor(
    kernel: str,
) -> None:
    params = {
        "logits": torch.tensor(
            [[0.5, -1.0, 2.0], [1.25, -0.75, 0.0]],
            dtype=torch.float64,
        )
    }
    vector = {
        "logits": torch.tensor(
            [[0.25, -0.5, 1.0], [-1.0, 0.75, 0.5]],
            dtype=torch.float64,
        )
    }

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return params["logits"]

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
        function_objectives={"model_output": function},
    )
    result = check(
        vpx.Candidate(
            "ggn",
            f"closed-form-{kernel}",
            ggn_closed_form_ce_kl_settings(kernel),
            admission_status="passed",
        ),
        {
            "loss_hessian": dense_softmax_ce_kl_loss_hessian(params["logits"]),
            "symmetry_vector": vector,
        },
        vector,
    )

    assert result.measurements["max_abs_diff"] <= 1e-12


@pytest.mark.parametrize(
    ("jvp_reuse", "cotangent_reuse"),
    [
        ("reuse_jvp", "reuse_output_cotangent"),
        ("reuse_jvp", "recompute_output_cotangent"),
        ("recompute_jvp", "reuse_output_cotangent"),
        ("recompute_jvp", "recompute_output_cotangent"),
    ],
)
def test_ggnvp_reuse_rows_match_dense_loss_hessian(
    jvp_reuse: str,
    cotangent_reuse: str,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            f"{jvp_reuse}-{cotangent_reuse}",
            ggn_reuse_settings(jvp_reuse, cotangent_reuse),
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    torch.testing.assert_close(result_map["w"], expected)


@pytest.mark.parametrize(
    ("jvp_path", "requires_forward_ad", "expected_calls"),
    [
        ("torch_func_jvp", True, 2),
        ("forward_ad_dual", True, 2),
        ("torch_func_linearize", False, 3),
    ],
)
def test_ggnvp_reuses_primal_from_jvp(
    jvp_path: str,
    requires_forward_ad: bool,
    expected_calls: int,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    calls = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"
        calls.append(None)

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    settings = {
        **ggn_settings(jvp_path),
        **torch_func_settings(requires_forward_ad=requires_forward_ad),
        "ggn.loss_hessian_path": "autodiff_loss_hvp",
        "ggn.loss_hessian_kernel": "dense_global",
    }
    result = factory(
        vpx.Candidate(
            "ggn",
            f"reuse-primal-{jvp_path}",
            settings,
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert len(calls) == expected_calls
    torch.testing.assert_close(result_map["w"], expected)


def test_ggnvp_jvp_hessian_vjp_does_not_call_dense_jacobian_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def forbidden_dense_jacobian(*_: Any, **__: Any) -> Any:
        message = "GGN JVP path called dense Jacobian helper"
        raise AssertionError(message)

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    monkeypatch.setattr(
        runtime_module,
        "_dense_jacobian_tree",
        forbidden_dense_jacobian,
    )
    monkeypatch.setattr(
        runtime_module,
        "_dense_jacobian_row_block",
        forbidden_dense_jacobian,
    )
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "jvp-hessian-vjp",
            ggn_dense_kernel_settings(),
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    torch.testing.assert_close(result_map["w"], expected)


def test_ggnvp_autodiff_loss_hvp_uses_output_space_ad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    calls = []
    original_jvp = runtime_module.torch.func.jvp

    def recording_jvp(
        func: Callable[..., Any],
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ) -> tuple[Any, Any]:
        if isinstance(primals[0], torch.Tensor) and isinstance(
            tangents[0],
            torch.Tensor,
        ):
            calls.append((primals[0].shape, tangents[0].shape))

        result = original_jvp(func, primals, tangents)

        return result[0], result[1]

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    monkeypatch.setattr(runtime_module.torch.func, "jvp", recording_jvp)
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "autodiff-loss-hvp",
            ggn_dense_kernel_settings(),
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert calls == [(torch.Size([2]), torch.Size([2]))]
    torch.testing.assert_close(result_map["w"], expected)


def test_ggnvp_rejects_backward_materialized_vjp_path() -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.tensor([[2.0, 0.5], [0.5, 4.0]], dtype=torch.float64)

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] + 2.0 * params["w"][1],
            3.0 * params["w"][0] - params["w"][1],
        ))

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    with pytest.raises(vp.MaterializationError, match=r"ggn\.vjp_path"):
        factory(
            vpx.Candidate(
                "ggn",
                "backward-vjp",
                {
                    **ggn_settings(
                        "torch_func_jvp",
                        vjp_path="backward_materialized_grad",
                    ),
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {"loss_hessian": loss_hessian},
            vector,
        )


def test_ggnvp_executes_intermediate_residency_at_jvp_and_cotangent_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    calls = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    def recording_residency(
        tensor: torch.Tensor,
        residency: object,
        key: str,
    ) -> torch.Tensor:
        moved = tensor.clone().requires_grad_(tensor.requires_grad)
        calls.append((key, residency, moved.detach().clone()))

        return moved

    monkeypatch.setattr(runtime_module, "_residency_tensor", recording_residency)
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "intermediate-residency",
            {
                **ggn_reuse_settings("reuse_jvp", "reuse_output_cotangent"),
                "memory.intermediate_residency": "cpu_staged",
            },
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert len(calls) == 2
    assert calls[0][0] == "memory.intermediate_residency"
    assert calls[1][0] == "memory.intermediate_residency"
    assert calls[0][1] == "cpu_staged"
    assert calls[1][1] == "cpu_staged"
    torch.testing.assert_close(calls[0][2], jacobian @ vector["w"])
    torch.testing.assert_close(calls[1][2], loss_hessian @ (jacobian @ vector["w"]))
    torch.testing.assert_close(result_map["w"], expected)


def test_ggnvp_executes_intermediate_transform_at_operator_part_boundaries() -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    calls = []

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    def intermediate_transform(tree: vpx.TensorTree) -> vpx.TensorTree:
        calls.append(tree_leaves(tree)[0].detach().clone())

        return tree

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
        intermediate_transform=intermediate_transform,
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "intermediate-transform",
            ggn_reuse_settings("reuse_jvp", "reuse_output_cotangent"),
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], jacobian @ vector["w"])
    torch.testing.assert_close(calls[1], loss_hessian @ (jacobian @ vector["w"]))
    torch.testing.assert_close(result_map["w"], expected)


def test_ggnvp_chunks_output_cotangent_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64))
    chunk_nonzeros = []
    original_vjp = runtime_module._run_ggnvp_vjp_by_path

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
            params["w"][0] * params["w"][1],
        ))

    def recording_vjp(
        execution: runtime_module.StandardExecution,
        tensor_function: Callable[[vpx.ParameterTree], vpx.TensorTree],
        output_cotangent: vpx.TensorTree,
    ) -> vpx.TensorTree:
        chunk_nonzeros.append(
            int(
                torch.count_nonzero(
                    runtime_module._flatten_vector(output_cotangent)
                ).item()
            )
        )

        return original_vjp(execution, tensor_function, output_cotangent)

    monkeypatch.setattr(runtime_module, "_run_ggnvp_vjp_by_path", recording_vjp)
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "chunked-output-cotangent",
            {
                **ggn_reuse_settings("reuse_jvp", "reuse_output_cotangent"),
                "chunk.output_cotangent_block_size": 1,
            },
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5], [-0.25, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert chunk_nonzeros == [1, 1, 1]
    torch.testing.assert_close(result_map["w"], expected)


def test_dense_ggnvp_executes_declared_row_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64))
    row_blocks = []
    original_row_block = runtime_module._dense_jacobian_row_block

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
            params["w"][0] * params["w"][1],
        ))

    def recording_row_block(
        output: torch.Tensor,
        parameter_leaves: tuple[torch.Tensor, ...],
        parameter_width: int,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        row_blocks.append((start, stop))

        return original_row_block(
            output,
            parameter_leaves,
            parameter_width,
            start,
            stop,
        )

    monkeypatch.setattr(
        runtime_module, "_dense_jacobian_row_block", recording_row_block
    )
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "row-batched-dense",
            {
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
                "batch.ggn_batch_size": 1,
            },
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5], [-0.25, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))
    result_map = tensor_mapping(result)

    assert row_blocks == [(0, 1), (1, 2), (2, 3), (0, 1), (1, 2), (2, 3)]
    torch.testing.assert_close(result_map["w"], expected)


@pytest.mark.parametrize(
    "settings",
    [
        {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
            "ggn.jvp_reuse": "reuse_jvp",
        },
        {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
            "ggn.cotangent_reuse": "reuse_output_cotangent",
        },
    ],
)
def test_dense_ggnvp_rejects_reuse_settings(settings: dict[str, object]) -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return params["w"]

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.MaterializationError, match="reuse"):
        factory(
            vpx.Candidate(
                "ggn",
                "dense-with-reuse",
                settings,
                admission_status="passed",
            ),
            {"loss_hessian": torch.eye(1, dtype=torch.float64)},
            vector,
        )


def test_fisher_vp_rejects_empirical_vmap_path() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    batch = {"x": torch.tensor([[1.0], [2.0]], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"

        return batch["x"].flatten()[:, None] * params["w"]

    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        function_objectives={"scores": score_rows},
    )

    with pytest.raises(vp.MaterializationError, match=r"fisher\.expectation_path"):
        factory(
            vpx.Candidate(
                "fisher",
                "vmap",
                {
                    **empirical_grad_settings("vmap_grad"),
                    **empirical_per_example_vmap_settings(),
                    **torch_func_settings(requires_forward_ad=False),
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()


def test_fisher_score_grad_paths_match_loop_result() -> None:
    params = {"w": torch.tensor([1.0, -1.0], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        "normalization": 3.0,
    }
    vector = {"w": torch.tensor([0.5, 0.25], dtype=torch.float64)}

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["normalization"] == pytest.approx(3.0)
        assert context.family == "fisher"
        x = batch["x"].reshape(-1)

        return x * params["w"][0] + 2.0 * x * params["w"][1]

    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        function_objectives={"scores": score_rows},
    )
    results = candidate_leaf_results(
        factory,
        "fisher",
        (
            (
                "loop",
                {
                    **fisher_settings(
                        "streaming_dot_accumulate",
                        score_grad_path="torch_autograd_grad_loop",
                    ),
                    "schedule.per_example": "loop",
                },
            ),
            (
                "torch-func",
                {
                    **fisher_settings(
                        "streaming_dot_accumulate",
                        score_grad_path="torch_func_grad",
                    ),
                    "schedule.per_example": "loop",
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
            (
                "backward",
                {
                    **fisher_settings(
                        "streaming_dot_accumulate",
                        score_grad_path="backward_materialized_grad",
                    ),
                    "schedule.per_example": "loop",
                },
            ),
            (
                "vmap",
                {
                    **fisher_settings(
                        "streaming_dot_accumulate",
                        score_grad_path="vmap_grad",
                    ),
                    **fisher_per_example_vmap_settings(batch_size=2),
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
        ),
        batch,
        vector,
    )
    expected = torch.tensor([14.0 / 3.0, 28.0 / 3.0], dtype=torch.float64)

    for result in results.values():
        assert torch.allclose(result, expected)


def test_streaming_fisher_family_accumulates_without_score_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
        "normalization": 3.0,
        "num_examples": 3,
    }
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family in {"fisher", "sampled", "empirical"}

        return params["w"][0] * batch["x"]

    def blocked_score_matrix_product(
        _: torch.Tensor,
        __: torch.Tensor,
        ___: float,
        ____: Mapping[str, object],
    ) -> torch.Tensor:
        message = "streaming row used full score matrix product"
        raise AssertionError(message)

    def blocked_gradient_matrix(_: runtime_module.StandardExecution) -> torch.Tensor:
        message = "streaming row built a stacked gradient matrix"
        raise AssertionError(message)

    monkeypatch.setattr(
        runtime_module,
        "_score_matrix_product",
        blocked_score_matrix_product,
    )
    monkeypatch.setattr(
        runtime_module,
        "_per_example_gradient_matrix",
        blocked_gradient_matrix,
    )
    factories = fisher_family_score_factories(params, score_rows)
    expected = torch.tensor([56.0 / 3.0], dtype=torch.float64)
    expected_sampled = torch.tensor([56.0 / 6.0], dtype=torch.float64)

    for family, settings, expected_result in (
        (
            "fisher",
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                "schedule.per_example": "loop",
            },
            expected,
        ),
        (
            "sampled",
            {
                **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
            },
            expected_sampled,
        ),
        (
            "empirical",
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
            },
            expected,
        ),
    ):
        result = run_passed_candidate(
            factories[family],
            family,
            "streaming",
            settings,
            batch,
            vector,
        )
        torch.testing.assert_close(tree_leaves(result)[0], expected_result)


def test_fisher_family_manual_per_example_schedule_runs_declared_subbatches() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
        "normalization": 3.0,
        "num_examples": 3,
    }
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        x_value = batch["x"]
        assert isinstance(x_value, torch.Tensor)
        calls.append((context.family, int(x_value.shape[0])))

        return params["w"][0] * x_value

    factories = fisher_family_score_factories(params, score_rows)

    for family, loop_settings, manual_settings in (
        (
            "fisher",
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                "schedule.per_example": "loop",
            },
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                **fisher_per_example_manual_settings(batch_size=2),
            },
        ),
        (
            "sampled",
            {
                **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
            },
            {
                **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                **fisher_per_example_manual_settings(batch_size=2),
            },
        ),
        (
            "empirical",
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
            },
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                **empirical_per_example_manual_settings(batch_size=2),
            },
        ),
    ):
        loop_result = run_passed_candidate(
            factories[family],
            family,
            "loop",
            loop_settings,
            batch,
            vector,
        )
        manual_result = run_passed_candidate(
            factories[family],
            family,
            "manual",
            manual_settings,
            batch,
            vector,
        )
        torch.testing.assert_close(
            tree_leaves(manual_result)[0],
            tree_leaves(loop_result)[0],
        )

    assert calls == [
        ("fisher", 3),
        ("fisher", 2),
        ("fisher", 1),
        ("sampled", 3),
        ("sampled", 2),
        ("sampled", 1),
        ("empirical", 3),
        ("empirical", 2),
        ("empirical", 1),
    ]


def test_manual_per_example_schedule_requires_declared_batch_size() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([2.0, 4.0], dtype=torch.float64),
        "normalization": 2.0,
    }
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def score_rows(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"

        return params["w"][0] * batch["x"]

    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        function_objectives={"scores": score_rows},
    )

    with pytest.raises(
        vp.MaterializationError,
        match=r"batch[.]fisher_sample_batch_size is required",
    ):
        factory(
            vpx.Candidate(
                "fisher",
                "manual",
                {
                    **fisher_settings(
                        "streaming_dot_accumulate",
                        score_grad_path="torch_autograd_grad_loop",
                    ),
                    "schedule.per_example": "manual_batch",
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()


def test_fisher_score_grad_path_axis_validates_torch_func_rows() -> None:
    registry = vpx.standard_axis_registry()
    admitted = registry.admit(
        vpx.Candidate(
            "fisher",
            "vmap",
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="vmap_grad",
                ),
                **fisher_per_example_vmap_settings(batch_size=2),
                **torch_func_settings(requires_forward_ad=False),
            },
        )
    )
    rejected = registry.admit(
        vpx.Candidate(
            "fisher",
            "missing-fields",
            fisher_settings(
                "streaming_dot_accumulate",
                score_grad_path="torch_func_grad",
            ),
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None
    assert "torch.func admission fields missing" in rejected.admission_error


def test_fisher_score_grad_path_is_required_only_for_streaming_rows() -> None:
    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"scores": square_function},
    )

    with pytest.raises(vp.MaterializationError, match=r"fisher\.score_grad_path"):
        factory(
            vpx.Candidate(
                "fisher",
                "missing-score-path",
                {
                    "fisher.expectation_path": "explicit_full_expectation_score_rows",
                    "fisher.accumulation": "streaming_dot_accumulate",
                },
                admission_status="passed",
            ),
            {"x": torch.tensor([1.0], dtype=torch.float64), "normalization": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.MaterializationError, match="not used"):
        factory(
            vpx.Candidate(
                "fisher",
                "dense-with-score-path",
                {
                    **fisher_settings("materialize_score_gradients"),
                    "fisher.score_grad_path": "torch_autograd_grad_loop",
                },
                admission_status="passed",
            ),
            {
                "score_gradients": torch.tensor([[1.0]], dtype=torch.float64),
                "normalization": 1.0,
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_fisher_vp_rejects_categorical_and_requires_score_reduction() -> None:
    with pytest.raises(vp.MaterializationError, match="GGNVP"):
        ops.fisher_vp(
            "fisher",
            "logits",
            aggregation="mean_per_example",
            distribution="categorical",
            label_policy="model_distribution",
            sample_space="classes",
            score_reduction="none",
            denominator="num_examples",
        )

    with pytest.raises(vp.MaterializationError, match="score_reduction"):
        ops.fisher_vp(
            "fisher",
            "scores",
            aggregation="mean_per_example",
            distribution="explicit_score_gradients",
            label_policy="explicit_scores",
            sample_space="terms",
            score_reduction="log_prob",
            denominator="batch_normalization",
        )


def test_inverse_metric_reference_check_records_inverse_residual() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "condition_number_max": 2.0,
        },
    )
    result = check(
        vpx.Candidate(
            "inverse",
            "row",
            inverse_metric_settings(),
            admission_status="passed",
        ),
        {"metric_matrix": torch.eye(2, dtype=torch.float64)},
        {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
    )

    assert result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert result.measurements["psd_violation"] == pytest.approx(0.0)
    assert result.measurements["condition_number_max"] == pytest.approx(1.0)


def test_inverse_metric_reference_check_uses_declared_damping() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.diag(torch.tensor([1.0, -0.1], dtype=torch.float64))
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.2,
    )
    thresholds = {
        "max_abs_diff": 1e-12,
        "max_rel_diff": 1e-12,
        "symmetry_max_abs_diff": 1e-12,
        "psd_violation": 1e-12,
        "inverse_residual": 1e-12,
        "damping_min": 0.1,
        "condition_number_max": 20.0,
    }
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds=thresholds,
    )
    result = check(
        vpx.Candidate(
            "inverse",
            "row",
            inverse_metric_settings(),
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )
    expected = torch.linalg.solve(
        matrix + 0.2 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )

    assert result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert result.measurements["psd_violation"] == pytest.approx(0.0)
    assert result.measurements["damping_min"] == pytest.approx(0.2)
    assert result.measurements["condition_number_max"] == pytest.approx(12.0)

    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    operation_result = factory(
        vpx.Candidate(
            "inverse",
            "row",
            inverse_metric_settings(),
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )()

    assert torch.allclose(tree_leaves(operation_result)[0], expected)

    failing_check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={**thresholds, "damping_min": 0.3},
    )

    with pytest.raises(vp.ReferenceFailedError, match="damping_min"):
        failing_check(
            vpx.Candidate(
                "inverse",
                "row",
                inverse_metric_settings(),
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            vector,
        )


def test_inverse_metric_inner_reference_check_records_inverse_residual() -> None:
    params = _zero_w_params(2)
    matrix = _float_rows(DENSE_2X2_ROWS)
    damping = 0.5
    left = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([3.0, -1.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.inverse_metric_inner(
            "inverse_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.4,
            "condition_number_max": 3.0,
        },
    )
    result = check(
        vpx.Candidate(
            "inverse_inner",
            "row",
            {
                "inverse_metric_inner.reduction_path": "solve_then_reduce",
                "inverse_metric_inner.multi_rhs": "single_column",
                **inverse_metric_settings(),
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        (left, right),
    )

    assert result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert result.measurements["damping_min"] == pytest.approx(damping)
    assert result.measurements["condition_number_max"] < 3.0


def test_inverse_metric_dense_direct_solve_paths_match_dense_solve() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = _float_rows(DENSE_2X2_ROWS)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "condition_number_max": 2.0,
        },
    )
    expected = torch.linalg.solve(matrix, vector["w"])

    for path in ("dense_solve", "cholesky_solve", "eigh_solve", "svd_solve"):
        candidate = vpx.Candidate(
            "inverse",
            path,
            inverse_metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, {"metric_matrix": matrix}, vector)()
        reference_result = check(candidate, {"metric_matrix": matrix}, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)


def test_inverse_metric_rejects_ill_conditioned_undamped_solve() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "condition_number_max": 1e6,
        },
    )

    with pytest.raises(vp.ReferenceFailedError, match="condition_number_max"):
        check(
            vpx.Candidate(
                "inverse",
                "ill-conditioned",
                inverse_metric_settings(),
                admission_status="passed",
            ),
            {
                "metric_matrix": torch.diag(
                    torch.tensor([1.0, 1e-8], dtype=torch.float64)
                )
            },
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        )


@pytest.mark.parametrize(
    "path",
    [
        "dense_solve",
        "cholesky_solve",
        "eigh_solve",
        "svd_solve",
        "factorized_solve",
        "blockwise_solve",
        "woodbury_low_rank_solve",
    ],
)
def test_inverse_metric_direct_solve_rejects_iteration_budget(path: str) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )
    factory = vpx.standard_operation_factory(operator, params=params, buffers={})

    with pytest.raises(vp.MaterializationError, match="conjugate_gradient"):
        factory(
            vpx.Candidate(
                "inverse",
                "direct-budget",
                {
                    **inverse_metric_settings(path),
                    "inverse_metric.iteration_budget": 2,
                },
                admission_status="passed",
            ),
            {"metric_matrix": torch.eye(2, dtype=torch.float64)},
            vector,
        )


def test_inverse_metric_refactor_each_rhs_is_executable() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = _float_rows(DENSE_2X2_ROWS)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
        ),
        params,
        {"metric_matrix": matrix},
        vector,
        {
            **inverse_metric_settings("cholesky_solve"),
            "inverse_metric.factor_reuse": "refactor_each_rhs",
        },
        "cholesky-refactor",
    )

    torch.testing.assert_close(
        tree_leaves(output)[0],
        torch.linalg.solve(matrix, vector["w"]),
    )


def test_inverse_metric_reuse_factor_across_rhs_requires_multi_rhs() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = _float_rows(DENSE_2X2_ROWS)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="multi_rhs"):
        factory(
            vpx.Candidate(
                "inverse",
                "cholesky-reuse",
                {
                    **inverse_metric_settings("cholesky_solve"),
                    "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
                    "vectorization.mode": "single_loop",
                    "vectorization.in_dims": {"w": 0},
                },
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            {"w": vector["w"].unsqueeze(0)},
        )


@pytest.mark.parametrize(
    ("multi_rhs", "expected_cholesky_calls"),
    [("single_column", 3), ("block", 1)],
)
def test_inverse_metric_multi_rhs_controls_cholesky_rhs_batching(
    monkeypatch: pytest.MonkeyPatch,
    multi_rhs: str,
    expected_cholesky_calls: int,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = _float_rows(DENSE_2X2_ROWS)
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [1.0, 2.0], [-0.5, 0.75]],
            dtype=torch.float64,
        )
    }
    calls = []
    original_cholesky = runtime_module.torch.linalg.cholesky

    def recording_cholesky(input_matrix: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(input_matrix.shape))

        return original_cholesky(input_matrix)

    monkeypatch.setattr(runtime_module.torch.linalg, "cholesky", recording_cholesky)
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )
    candidate = vpx.standard_axis_registry().admit(
        vpx.Candidate(
            "inverse",
            f"cholesky-{multi_rhs}",
            {
                **inverse_metric_settings("cholesky_solve"),
                "inverse_metric.multi_rhs": multi_rhs,
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
        )
    )
    assert candidate.admission_status == "passed"
    factory = vpx.standard_operation_factory(operator, params=params, buffers={})
    output = factory(candidate, {"metric_matrix": matrix}, vector)()
    expected = torch.linalg.solve(matrix, vector["w"].T).T

    torch.testing.assert_close(tree_leaves(output)[0], expected)
    assert calls == [(2, 2)] * expected_cholesky_calls


def test_inverse_metric_reuse_factor_across_rhs_solves_vectorized_rhs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = _float_rows(DENSE_2X2_ROWS)
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [1.0, 2.0], [-0.5, 0.75]],
            dtype=torch.float64,
        )
    }
    calls = []
    original_cholesky = runtime_module.torch.linalg.cholesky
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )

    def recording_cholesky(input_matrix: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(input_matrix.shape))

        return original_cholesky(input_matrix)

    monkeypatch.setattr(runtime_module.torch.linalg, "cholesky", recording_cholesky)
    output = standard_operation_result(
        operator,
        params,
        {"metric_matrix": matrix},
        vector,
        {
            **inverse_metric_settings("cholesky_solve"),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "cholesky-reuse-vectorized",
    )
    expected = torch.linalg.solve(matrix, vector["w"].T).T

    torch.testing.assert_close(tree_leaves(output)[0], expected)
    assert calls == [(2, 2)]


def test_diagonal_inverse_metric_reuse_factor_across_rhs() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [1.0, 2.0], [-0.5, 0.75]],
            dtype=torch.float64,
        )
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
            damping=0.25,
        ),
        params,
        {"metric_diagonal": diagonal},
        vector,
        {
            **inverse_metric_settings("factorized_solve"),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "diagonal-reuse",
    )

    torch.testing.assert_close(
        tree_leaves(output)[0],
        vector["w"] / (diagonal["w"] + 0.25),
    )


def test_block_inverse_metric_reuse_factor_across_rhs() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([[0.25], [1.0], [-0.5]], dtype=torch.float64),
        "b": torch.tensor(
            [[-0.75, 0.5], [2.0, 1.5], [0.75, -1.0]],
            dtype=torch.float64,
        ),
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "blocks",
            aggregation="sum",
            representation=block_metric_representation(),
            damping=0.25,
        ),
        params,
        {"metric_blocks": blocks},
        vector,
        {
            **inverse_metric_settings("blockwise_solve"),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"a": 0, "b": 0},
        },
        "block-reuse",
    )
    expected_flat = torch.linalg.solve(
        torch.block_diag(*blocks) + 0.25 * torch.eye(3, dtype=torch.float64),
        torch.cat((vector["a"], vector["b"]), dim=1).T,
    ).T
    output_map = tensor_mapping(output)

    torch.testing.assert_close(output_map["a"], expected_flat[:, :1])
    torch.testing.assert_close(output_map["b"], expected_flat[:, 1:])


@pytest.mark.parametrize("path", ["factorized_solve", "woodbury_low_rank_solve"])
def test_low_rank_inverse_metric_reuse_factor_across_rhs(path: str) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = {
        "basis": torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        "diagonal": torch.tensor([4.0, 5.0], dtype=torch.float64),
    }
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [1.0, 2.0], [-0.5, 0.75]],
            dtype=torch.float64,
        )
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
            damping=0.25,
        ),
        params,
        {"low_rank_factors": factors},
        vector,
        {
            **inverse_metric_settings(path),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "low-rank-reuse",
    )
    dense_matrix = factors["basis"] @ factors["basis"].T + torch.diag(
        factors["diagonal"]
    )
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"].T,
    ).T

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_kfac_inverse_metric_reuse_factor_across_rhs() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = KFACMetricData.factors()
    vector = {
        "w": torch.tensor(
            [
                [[0.25, -0.75], [0.5, 1.25]],
                [[1.0, 0.5], [-0.25, 0.75]],
                [[-0.5, 0.25], [1.5, -1.0]],
            ],
            dtype=torch.float64,
        )
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "kfac",
            aggregation="sum",
            representation=kfac_metric_representation(),
            damping=0.25,
        ),
        params,
        {"kfac_factors": factors},
        vector,
        {
            **inverse_metric_settings("factorized_solve"),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "kfac-reuse",
    )
    dense_matrix = torch.kron(factors["w_left"], factors["w_right"])
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(4, dtype=torch.float64),
        vector["w"].reshape(3, -1).T,
    ).T.reshape(3, 2, 2)

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_ggn_inverse_metric_reuse_factor_across_rhs_matches_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = GGNMetricData.factors()
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [1.0, 2.0], [-0.5, 0.75]],
            dtype=torch.float64,
        )
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
            damping=0.25,
        ),
        params,
        {"ggn_factors": factors},
        vector,
        {
            **inverse_metric_settings("factorized_solve"),
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "ggn-reuse",
    )

    jacobian = factors["jacobian"]
    loss_hessian = factors["loss_hessian"]
    dense_matrix = jacobian.T @ loss_hessian @ jacobian
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"].T,
    ).T

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_cg_inverse_metric_reuse_factor_across_rhs() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    vector = {
        "w": torch.tensor(
            [[0.25, -0.75], [0.0, 0.0], [1.0, 2.0]],
            dtype=torch.float64,
        )
    }
    output = standard_operation_result(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
        ),
        params,
        {"metric_matrix": matrix},
        vector,
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 2,
            "inverse_metric.preconditioner": "none",
            "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
            "inverse_metric.multi_rhs": "block",
            "metric.multiply_path": "dense_matmul",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "cg-reuse",
    )
    expected = torch.linalg.solve(matrix, vector["w"].T).T

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_diagonal_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "diagonal",
        aggregation="sum",
        representation=diagonal_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    expected = diagonal["w"] * vector["w"]
    batch = {"metric_diagonal": diagonal}

    for path in ("factorized_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_metric_inner_dense_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    left = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([3.0, -1.0], dtype=torch.float64)}
    operator = ops.metric_inner(
        "metric_inner",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    expected = left["w"] @ (matrix @ right["w"])

    multiply_result = factory(
        vpx.Candidate(
            "metric_inner",
            "multiply",
            {
                "metric_inner.reduction_path": "multiply_then_reduce",
                "metric_inner.multi_rhs": "single_column",
                **metric_settings("dense_matmul"),
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        (left, right),
    )()
    sqrt_result = factory(
        vpx.Candidate(
            "metric_inner",
            "sqrt",
            {
                "metric_inner.reduction_path": "sqrt_apply_reduce",
                "metric_inner.multi_rhs": "single_column",
                "sqrt_metric.factor_path": "cholesky_factor",
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        (left, right),
    )()

    torch.testing.assert_close(multiply_result, expected)
    torch.testing.assert_close(sqrt_result, expected)


def test_sqrt_metric_cholesky_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    damping = 0.5
    vector = {"w": torch.tensor([3.0, -1.0], dtype=torch.float64)}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"metric_matrix": matrix},
        settings={"sqrt_metric.factor_path": "cholesky_factor"},
        sqrt_vector=vector,
        inverse_vector=vector,
    )
    expected_sqrt = torch.linalg.cholesky(matrix) @ vector["w"]
    inverse_matrix = torch.linalg.inv(
        matrix + damping * torch.eye(2, dtype=torch.float64)
    )
    expected_inverse_sqrt = torch.linalg.cholesky(inverse_matrix) @ vector["w"]

    torch.testing.assert_close(tensor_mapping(sqrt_result)["w"], expected_sqrt)
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        expected_inverse_sqrt,
    )


def test_sqrt_metric_eigenbasis_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[5.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    damping = 0.25
    vector = {"w": torch.tensor([0.5, -1.5], dtype=torch.float64)}
    settings = {"sqrt_metric.factor_path": "eigenbasis_factor"}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"metric_matrix": matrix},
        settings=settings,
        sqrt_vector=vector,
        inverse_vector=vector,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ vector["w"]
    inverse_matrix = torch.linalg.inv(
        matrix + damping * torch.eye(2, dtype=torch.float64)
    )
    inverse_eigenvalues, inverse_eigenvectors = torch.linalg.eigh(inverse_matrix)
    expected_inverse_sqrt = (
        inverse_eigenvectors @ torch.diag(torch.sqrt(inverse_eigenvalues)) @ vector["w"]
    )

    torch.testing.assert_close(tensor_mapping(sqrt_result)["w"], expected_sqrt)
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        expected_inverse_sqrt,
    )


def test_sqrt_metric_closed_form_diagonal_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 9.0], dtype=torch.float64)}
    damping = 0.75
    vector = {"w": torch.tensor([0.5, -1.5], dtype=torch.float64)}
    settings = {"sqrt_metric.factor_path": "closed_form_factor_square_root"}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"metric_diagonal": diagonal},
        settings=settings,
        sqrt_vector=vector,
        inverse_vector=vector,
    )

    torch.testing.assert_close(
        tensor_mapping(sqrt_result)["w"],
        torch.sqrt(diagonal["w"]) * vector["w"],
    )
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        torch.rsqrt(diagonal["w"] + damping) * vector["w"],
    )


def test_sqrt_metric_closed_form_low_rank_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    damping = 0.75
    latent = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)
    vector = {"w": torch.tensor([0.5, -1.5], dtype=torch.float64)}
    settings = {"sqrt_metric.factor_path": "closed_form_factor_square_root"}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"low_rank_factors": factors},
        settings=settings,
        sqrt_vector=latent,
        inverse_vector=vector,
    )
    basis = factors["basis"]
    diagonal = factors["diagonal"]
    rank = basis.shape[1]
    expected_sqrt = basis @ latent[:rank] + torch.sqrt(diagonal) * latent[rank:]
    base_diagonal = diagonal + damping
    scaled_basis = basis / torch.sqrt(base_diagonal).unsqueeze(1)
    core = symmetric_square_root(
        torch.linalg.inv(
            torch.eye(2, dtype=torch.float64) + scaled_basis @ scaled_basis.T
        )
    )
    expected_inverse_sqrt = (core @ vector["w"]) / torch.sqrt(base_diagonal)

    torch.testing.assert_close(tensor_mapping(sqrt_result)["w"], expected_sqrt)
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        expected_inverse_sqrt,
    )


def test_metric_inner_low_rank_sqrt_reduce_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    damping = 0.25
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    settings = {
        "sqrt_metric.factor_path": "closed_form_factor_square_root",
    }
    metric_result, inverse_result = metric_inner_single_column_pair_results(
        ops.metric_inner(
            "metric_inner",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
        ),
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"low_rank_factors": factors},
        left=left,
        right=right,
        metric_reduction_path="sqrt_apply_reduce",
        inverse_reduction_path="sqrt_apply_reduce",
        shared_settings=settings,
    )
    matrix = factors["basis"] @ factors["basis"].T + torch.diag(factors["diagonal"])
    damped = matrix + damping * torch.eye(2, dtype=torch.float64)

    torch.testing.assert_close(metric_result, left["w"] @ (matrix @ right["w"]))
    torch.testing.assert_close(
        inverse_result,
        left["w"] @ torch.linalg.solve(damped, right["w"]),
    )


def test_sqrt_metric_closed_form_ggn_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = GGNMetricData.factors()
    damping = 0.75
    latent = torch.tensor([0.5, -1.25], dtype=torch.float64)
    vector = {"w": torch.tensor([0.5, -1.5], dtype=torch.float64)}
    settings = {"sqrt_metric.factor_path": "closed_form_factor_square_root"}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"ggn_factors": factors},
        settings=settings,
        sqrt_vector=latent,
        inverse_vector=vector,
    )
    jacobian = factors["jacobian"]
    loss_root = symmetric_square_root(factors["loss_hessian"])
    expected_sqrt = jacobian.T @ (loss_root @ latent)
    factor_basis = (loss_root @ jacobian).T
    scaled_basis = factor_basis / torch.sqrt(torch.tensor(damping, dtype=torch.float64))
    core = symmetric_square_root(
        torch.linalg.inv(
            torch.eye(2, dtype=torch.float64) + scaled_basis @ scaled_basis.T
        )
    )
    expected_inverse_sqrt = (
        core @ vector["w"] / torch.sqrt(torch.tensor(damping, dtype=torch.float64))
    )

    torch.testing.assert_close(tensor_mapping(sqrt_result)["w"], expected_sqrt)
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        expected_inverse_sqrt,
    )


def test_metric_inner_ggn_sqrt_reduce_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = GGNMetricData.factors()
    damping = 0.25
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    settings = {
        "sqrt_metric.factor_path": "closed_form_factor_square_root",
    }
    metric_result, inverse_result = metric_inner_single_column_pair_results(
        ops.metric_inner(
            "metric_inner",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
        ),
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"ggn_factors": factors},
        left=left,
        right=right,
        metric_reduction_path="sqrt_apply_reduce",
        inverse_reduction_path="sqrt_apply_reduce",
        shared_settings=settings,
    )
    matrix = factors["jacobian"].T @ factors["loss_hessian"] @ factors["jacobian"]
    damped = matrix + damping * torch.eye(2, dtype=torch.float64)

    torch.testing.assert_close(metric_result, left["w"] @ (matrix @ right["w"]))
    torch.testing.assert_close(
        inverse_result,
        left["w"] @ torch.linalg.solve(damped, right["w"]),
    )


def test_sqrt_metric_matrix_free_lanczos_rejects_dense_backing() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    vector = {"w": torch.tensor([0.5, -1.5], dtype=torch.float64)}
    settings = {
        "sqrt_metric.factor_path": "matrix_free_lanczos",
        "sqrt_metric.lanczos_iterations": 8,
    }
    factory = vpx.standard_operation_factory(
        ops.sqrt_metric(
            "sqrt_metric",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
        ),
        params=params,
        buffers={},
    )

    with pytest.raises(
        vp.MaterializationError,
        match="metric representation kind is not supported by path",
    ):
        factory(
            vpx.Candidate(
                "sqrt_metric",
                "matrix-free-lanczos",
                settings,
                admission_status="passed",
            ),
            {"metric_diagonal": {"w": torch.tensor([4.0, 9.0], dtype=torch.float64)}},
            vector,
        )


def test_inverse_sqrt_metric_matrix_free_lanczos_tol_checks_residual() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[5.0, 2.0], [2.0, 2.0]], dtype=torch.float64)
    vector = {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}
    settings = {
        "sqrt_metric.factor_path": "matrix_free_lanczos",
        "sqrt_metric.lanczos_iterations": 1,
        "metric.multiply_path": "streaming_multiply",
        "metric.accumulation": "streaming",
    }
    calls = []

    def curvature(
        batch: vpx.Batch,
        input_vector: vpx.TensorTree,
    ) -> vpx.TensorTree:
        assert batch == {}
        calls.append(tree_leaves(input_vector)[0].clone())

        return {"w": matrix @ tensor_mapping(input_vector)["w"]}

    def bound_factory(operator: Any) -> Any:
        runtime = runtime_module.standard_runtime_with_matrix_free_bindings(
            vpx.standard_runtime_config(
                operator,
                params=params,
                buffers={},
                candidates=(),
                thresholds={
                    "max_abs_diff": 1e-12,
                    "max_rel_diff": 1e-12,
                    "inverse_residual": 1e-12,
                },
                objective_signature={"inverse_sqrt": "matrix-free-lanczos-tol"},
                axis_registry=None,
            ),
            bindings={"curvature": curvature},
            binding_signature={"curvature": {"kind": "matrix_free_metric"}},
        )

        return runtime.operation_factory

    passing_operator = ops.inverse_sqrt_metric(
        "inverse_sqrt_metric",
        "metric",
        aggregation="sum",
        representation={"kind": "matrix_free", "operator": "curvature"},
        damping=0.5,
        tol=3.0,
    )
    result = bound_factory(passing_operator)(
        passed_candidate("inverse_sqrt_metric", "passing-lanczos-tol", settings),
        {},
        vector,
    )()
    expected = vector["w"] / torch.sqrt(torch.tensor(5.5, dtype=torch.float64))

    assert calls
    torch.testing.assert_close(tensor_mapping(result)["w"], expected)

    failing_operator = ops.inverse_sqrt_metric(
        "inverse_sqrt_metric",
        "metric",
        aggregation="sum",
        representation={"kind": "matrix_free", "operator": "curvature"},
        damping=0.5,
        tol=1e-12,
    )

    with pytest.raises(
        vp.MaterializationError,
        match="Lanczos residual exceeded tol",
    ):
        bound_factory(failing_operator)(
            passed_candidate("inverse_sqrt_metric", "failing-lanczos-tol", settings),
            {},
            vector,
        )()


def test_metric_inner_eigenbasis_sqrt_reduce_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[5.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    damping = 0.25
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    settings = {"sqrt_metric.factor_path": "eigenbasis_factor"}
    metric_result, inverse_result = metric_inner_single_column_pair_results(
        ops.metric_inner(
            "metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"metric_matrix": matrix},
        left=left,
        right=right,
        metric_reduction_path="sqrt_apply_reduce",
        inverse_reduction_path="sqrt_apply_reduce",
        shared_settings=settings,
    )
    inverse_matrix = matrix + damping * torch.eye(2, dtype=torch.float64)

    torch.testing.assert_close(metric_result, left["w"] @ (matrix @ right["w"]))
    torch.testing.assert_close(
        inverse_result,
        left["w"] @ torch.linalg.solve(inverse_matrix, right["w"]),
    )


def test_metric_inner_diagonal_factored_gram_paths_match_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 9.0], dtype=torch.float64)}
    damping = 0.75
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    batch = {"metric_diagonal": diagonal}
    metric_result = metric_inner_factored_gram_result(
        ops.metric_inner(
            "metric_inner",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
        ),
        params=params,
        batch=batch,
        left=left,
        right=right,
    )
    inverse_result = metric_inner_factored_gram_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch=batch,
        left=left,
        right=right,
    )

    torch.testing.assert_close(metric_result, left["w"] @ (diagonal["w"] * right["w"]))
    torch.testing.assert_close(
        inverse_result,
        left["w"] @ (right["w"] / (diagonal["w"] + damping)),
    )


def test_metric_inner_low_rank_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    result = metric_inner_factored_gram_result(
        ops.metric_inner(
            "metric_inner",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
        ),
        params=params,
        batch={"low_rank_factors": factors},
        left=left,
        right=right,
    )
    matrix = factors["basis"] @ factors["basis"].T + torch.diag(factors["diagonal"])

    torch.testing.assert_close(result, left["w"] @ (matrix @ right["w"]))


def test_inverse_metric_inner_low_rank_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    damping = 0.25
    factors = LowRankMetricData.factors()
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    result = metric_inner_factored_gram_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "low_rank",
            aggregation="sum",
            representation=low_rank_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"low_rank_factors": factors},
        left=left,
        right=right,
    )
    matrix = factors["basis"] @ factors["basis"].T + torch.diag(factors["diagonal"])
    damped = matrix + damping * torch.eye(2, dtype=torch.float64)

    torch.testing.assert_close(
        result,
        left["w"] @ torch.linalg.solve(damped, right["w"]),
    )


def test_metric_inner_kfac_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = KFACMetricData.factors()
    left = {"w": torch.tensor([[1.5, -0.5], [0.75, 0.25]], dtype=torch.float64)}
    right = {"w": torch.tensor([[0.25, 2.0], [-1.0, 0.5]], dtype=torch.float64)}
    result = metric_inner_factored_gram_result(
        ops.metric_inner(
            "metric_inner",
            "kfac",
            aggregation="sum",
            representation=kfac_metric_representation(),
        ),
        params=params,
        batch={"kfac_factors": factors},
        left=left,
        right=right,
    )
    matrix = torch.kron(factors["w_left"], factors["w_right"])

    torch.testing.assert_close(
        result,
        flatten_tree(left) @ (matrix @ flatten_tree(right)),
    )


def test_inverse_metric_inner_kfac_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    damping = 0.25
    factors = KFACMetricData.factors()
    left = {"w": torch.tensor([[1.5, -0.5], [0.75, 0.25]], dtype=torch.float64)}
    right = {"w": torch.tensor([[0.25, 2.0], [-1.0, 0.5]], dtype=torch.float64)}
    result = metric_inner_factored_gram_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "kfac",
            aggregation="sum",
            representation=kfac_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"kfac_factors": factors},
        left=left,
        right=right,
    )
    matrix = torch.kron(factors["w_left"], factors["w_right"])
    damped = matrix + damping * torch.eye(4, dtype=torch.float64)

    torch.testing.assert_close(
        result,
        flatten_tree(left) @ torch.linalg.solve(damped, flatten_tree(right)),
    )


def test_metric_inner_ggn_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = GGNMetricData.factors()
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    result = metric_inner_factored_gram_result(
        ops.metric_inner(
            "metric_inner",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
        ),
        params=params,
        batch={"ggn_factors": factors},
        left=left,
        right=right,
    )
    matrix = factors["jacobian"].T @ factors["loss_hessian"] @ factors["jacobian"]

    torch.testing.assert_close(result, left["w"] @ (matrix @ right["w"]))


def test_inverse_metric_inner_ggn_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    factors = GGNMetricData.factors()
    left = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    right = {"w": torch.tensor([0.25, 2.0], dtype=torch.float64)}
    damping = 0.25
    result = metric_inner_factored_gram_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "ggn",
            aggregation="sum",
            representation=ggn_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch={"ggn_factors": factors},
        left=left,
        right=right,
    )
    matrix = factors["jacobian"].T @ factors["loss_hessian"] @ factors["jacobian"]
    expected = left["w"] @ torch.linalg.solve(
        matrix + damping * torch.eye(2, dtype=torch.float64),
        right["w"],
    )

    torch.testing.assert_close(result, expected)


def test_metric_inner_requires_declared_multi_rhs_mode() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    left = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([3.0, -1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.metric_inner(
            "metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match=r"metric_inner\.multi_rhs"):
        factory(
            vpx.Candidate(
                "metric_inner",
                "missing-multi-rhs",
                {
                    "metric_inner.reduction_path": "multiply_then_reduce",
                    "metric.multiply_path": "dense_matmul",
                },
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            (left, right),
        )()


def test_metric_inner_block_rhs_matches_single_column_gram() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    left = {"w": torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)}
    right = {"w": torch.tensor([[3.0, -1.0], [2.0, 1.0]], dtype=torch.float64)}
    result = standard_operation_result(
        ops.metric_inner(
            "metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params,
        {"metric_matrix": matrix},
        (left, right),
        {
            "metric_inner.reduction_path": "multiply_then_reduce",
            "metric.multiply_path": "dense_matmul",
            "metric_inner.multi_rhs": "block",
            "vectorization.mode": "single_loop",
            "vectorization.in_dims": {"w": 0},
        },
        "block",
    )
    expected = left["w"] @ (right["w"] @ matrix.T).T

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("mode_settings", "candidate_id"),
    [
        (
            BLOCK_MANUAL_VECTOR_SETTINGS,
            "block-manual-batch",
        ),
        (
            BLOCK_VMAP_VECTOR_SETTINGS,
            "block-vmap",
        ),
    ],
)
def test_metric_inner_block_rhs_vectorization_modes_match_reference(
    mode_settings: Mapping[str, object],
    candidate_id: str,
) -> None:
    params = _zero_w_params(2)
    matrix = _float_rows(DENSE_2X2_ROWS)
    left = _vector_rows("w", INNER_LEFT_BLOCK_ROWS)
    right = _vector_rows("w", INNER_RIGHT_BLOCK_ROWS)
    result = standard_operation_result(
        ops.metric_inner(
            "metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params,
        {"metric_matrix": matrix},
        (left, right),
        {
            "metric_inner.reduction_path": "multiply_then_reduce",
            "metric.multiply_path": "dense_matmul",
            "metric_inner.multi_rhs": "block",
            **mode_settings,
        },
        candidate_id,
    )
    expected = left["w"] @ (matrix @ right["w"])

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("settings", "candidate_id"),
    [
        (
            {"metric_inner.reduction_path": "factored_gram"},
            "block-vmap-factored-gram",
        ),
        (
            {
                "metric_inner.reduction_path": "sqrt_apply_reduce",
                "sqrt_metric.factor_path": "closed_form_factor_square_root",
            },
            "block-vmap-sqrt-apply-reduce",
        ),
    ],
)
def test_metric_inner_vmap_block_factor_paths_match_reference(
    settings: Mapping[str, object],
    candidate_id: str,
) -> None:
    params = _zero_w_params(2)
    diagonal = {"w": torch.tensor([2.0, 5.0], dtype=torch.float64)}
    left = _vector_rows("w", INNER_LEFT_BLOCK_ROWS)
    right = _vector_rows("w", INNER_RIGHT_BLOCK_ROWS)
    result = standard_operation_result(
        ops.metric_inner(
            "metric_inner",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
        ),
        params,
        {"metric_diagonal": diagonal},
        (left, right),
        {
            **settings,
            "metric_inner.multi_rhs": "block",
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": ({"w": 0}, {"w": 1}),
            **torch_func_settings(requires_forward_ad=False),
        },
        candidate_id,
    )
    expected = left["w"] @ (diagonal["w"].reshape(-1, 1) * right["w"])

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("mode_settings", "candidate_id"),
    [
        (
            BLOCK_MANUAL_VECTOR_SETTINGS,
            "block-manual-batch",
        ),
        (
            BLOCK_VMAP_VECTOR_SETTINGS,
            "block-vmap",
        ),
    ],
)
def test_inverse_metric_inner_block_rhs_vectorization_modes_match_reference(
    mode_settings: Mapping[str, object],
    candidate_id: str,
) -> None:
    params = _zero_w_params(2)
    matrix = _float_rows(DENSE_2X2_ROWS)
    damping = 0.5
    left = _vector_rows("w", INNER_LEFT_BLOCK_ROWS)
    right = _vector_rows("w", INNER_RIGHT_BLOCK_ROWS)
    result = standard_operation_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params,
        {"metric_matrix": matrix},
        (left, right),
        {
            "inverse_metric_inner.reduction_path": "solve_then_reduce",
            "inverse_metric_inner.multi_rhs": "block",
            "inverse_metric.solve_path": "dense_solve",
            **mode_settings,
        },
        candidate_id,
    )
    damped = matrix + damping * torch.eye(2, dtype=torch.float64)
    expected = left["w"] @ torch.linalg.solve(damped, right["w"])

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("settings", "candidate_id"),
    [
        (
            {
                "inverse_metric_inner.reduction_path": "factored_gram",
            },
            "block-vmap-inverse-factored-gram",
        ),
        (
            {
                "inverse_metric_inner.reduction_path": "sqrt_apply_reduce",
                "sqrt_metric.factor_path": "closed_form_factor_square_root",
            },
            "block-vmap-inverse-sqrt-apply-reduce",
        ),
    ],
)
def test_inverse_metric_inner_vmap_block_factor_paths_match_reference(
    settings: Mapping[str, object],
    candidate_id: str,
) -> None:
    params = _zero_w_params(2)
    diagonal = {"w": torch.tensor([2.0, 5.0], dtype=torch.float64)}
    damping = 0.5
    left = _vector_rows("w", INNER_LEFT_BLOCK_ROWS)
    right = _vector_rows("w", INNER_RIGHT_BLOCK_ROWS)
    result = standard_operation_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "diagonal",
            aggregation="sum",
            representation=diagonal_metric_representation(),
            damping=damping,
        ),
        params,
        {"metric_diagonal": diagonal},
        (left, right),
        {
            **settings,
            "inverse_metric_inner.multi_rhs": "block",
            "vectorization.mode": "vmap",
            "vectorization.vmap_chunk_size": 2,
            "vectorization.in_dims": ({"w": 0}, {"w": 1}),
            **torch_func_settings(requires_forward_ad=False),
        },
        candidate_id,
    )
    expected = left["w"] @ (right["w"] / (diagonal["w"] + damping).reshape(-1, 1))

    torch.testing.assert_close(result, expected)


def test_inverse_metric_inner_solve_path_matches_reference() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    damping = 0.5
    left = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([3.0, -1.0], dtype=torch.float64)}
    result = standard_operation_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=damping,
        ),
        params,
        {"metric_matrix": matrix},
        (left, right),
        {
            "inverse_metric_inner.reduction_path": "solve_then_reduce",
            "inverse_metric_inner.multi_rhs": "single_column",
            "inverse_metric.solve_path": "dense_solve",
        },
        "solve",
    )
    expected = left["w"] @ torch.linalg.solve(
        matrix + damping * torch.eye(2, dtype=torch.float64),
        right["w"],
    )

    torch.testing.assert_close(result, expected)


def test_metric_inner_norm_requires_sqrt_apply_reduce() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    operator = ops.metric_inner(
        "metric_inner",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        as_norm=True,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="sqrt_apply_reduce"):
        factory(
            passed_candidate(
                "metric_inner",
                "bad-norm",
                {
                    "metric_inner.reduction_path": "multiply_then_reduce",
                    **metric_settings("dense_matmul"),
                },
            ),
            {"metric_matrix": matrix},
            (vector, vector),
        )()

    result = run_passed_candidate(
        factory,
        "metric_inner",
        "norm",
        {
            "metric_inner.reduction_path": "sqrt_apply_reduce",
            "metric_inner.multi_rhs": "single_column",
            "sqrt_metric.factor_path": "cholesky_factor",
        },
        {"metric_matrix": matrix},
        (vector, vector),
    )

    assert isinstance(result, torch.Tensor)
    assert result.item() >= 0.0


def test_dtype_accumulation_reaches_metric_multiply_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.tensor(
        [[4.0, 0.5], [0.5, 5.0]],
        dtype=torch.float64,
    )
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    original = runtime_module._accumulation_tensor
    calls = []

    def recording_accumulation_tensor(
        tensor: torch.Tensor,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        result = original(tensor, settings)

        if settings.get("dtype.accumulation") == "fp32":
            calls.append((tensor.dtype, result.dtype))

        return result

    monkeypatch.setattr(
        runtime_module,
        "_accumulation_tensor",
        recording_accumulation_tensor,
    )
    candidate = passed_candidate(
        "metric",
        "fp32-accumulation",
        {
            **metric_settings("dense_matmul"),
            "dtype.accumulation": "fp32",
        },
    )
    output = factory(candidate, {"metric_matrix": matrix}, vector)()

    assert torch.allclose(tree_leaves(output)[0], matrix.float() @ vector["w"].float())
    assert calls
    assert all(output_dtype == torch.float32 for _, output_dtype in calls)


def test_standard_runtime_executes_foreach_vector_ops_for_diagonal_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    calls = []
    original = torch._foreach_mul

    def foreach_mul(
        left: tuple[torch.Tensor, ...] | list[torch.Tensor],
        right: tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        calls.append((left, right))

        return original(left, right)

    monkeypatch.setattr(torch, "_foreach_mul", foreach_mul)
    operator = ops.metric(
        "metric",
        "diagonal",
        aggregation="sum",
        representation=diagonal_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = passed_candidate(
        "metric",
        "foreach-diagonal",
        {
            **metric_settings("factorized_multiply"),
            "layout.vector_ops": "foreach",
        },
    )
    output = factory(candidate, {"metric_diagonal": diagonal}, vector)()

    assert torch.allclose(tree_leaves(output)[0], diagonal["w"] * vector["w"])
    assert len(calls) == 1


def test_non_dense_metric_paths_require_accumulation() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "diagonal",
        aggregation="sum",
        representation=diagonal_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = passed_candidate(
        "metric",
        "missing-accumulation",
        {"metric.multiply_path": "factorized_multiply"},
    )

    with pytest.raises(
        vp.MaterializationError, match=r"metric\.accumulation is required"
    ):
        factory(candidate, {"metric_diagonal": diagonal}, vector)


def test_metric_accumulation_must_match_metric_path() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = passed_candidate(
        "metric",
        "wrong-accumulation",
        {
            "metric.multiply_path": "factorized_multiply",
            "metric.accumulation": "streaming",
        },
    )

    with pytest.raises(vp.MaterializationError, match="must be materialized_blocks"):
        factory(candidate, {"low_rank_factors": factors}, vector)

    candidate = passed_candidate(
        "metric",
        "wrong-streaming-accumulation",
        {
            "metric.multiply_path": "streaming_multiply",
            "metric.accumulation": "materialized_blocks",
        },
    )

    with pytest.raises(vp.MaterializationError, match="must be streaming"):
        factory(candidate, {"low_rank_factors": factors}, vector)


def test_standard_runtime_executes_metric_factor_dtype_axis() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    result = run_passed_candidate(
        factory,
        "metric",
        "metric-factor-dtype",
        {
            **metric_settings("factorized_multiply"),
            "dtype.vector": "fp32",
            "dtype.intermediate": "bf16",
            "dtype.metric_factor": "fp32",
        },
        {"low_rank_factors": factors},
        vector,
    )
    result_tensor = tree_leaves(result)[0]

    assert result_tensor.dtype == torch.bfloat16
    torch.testing.assert_close(
        result_tensor,
        torch.tensor([-0.25, -6.25], dtype=torch.bfloat16),
    )


def test_standard_runtime_executes_vector_residency_axis() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "vector-residency",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.vector_residency": "cpu_staged",
        },
        params,
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_executes_pinned_vector_residency() -> None:
    try:
        torch.empty(1).pin_memory()
    except RuntimeError as error:
        pytest.skip(str(error))

    calls = []
    original = runtime_module._residency_tensor
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}

    def recording_residency(
        tensor: torch.Tensor,
        residency: object,
        key: str,
    ) -> torch.Tensor:
        calls.append((key, residency))

        return original(tensor, residency, key)

    factory = quadratic_hvp_factory(params)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runtime_module, "_residency_tensor", recording_residency)
        result = run_passed_candidate(
            factory,
            "hvp",
            "vector-residency-pinned",
            {
                "hvp.path": "reverse_over_reverse",
                "memory.vector_residency": "cpu_pinned",
            },
            {"scale": 1.0},
            vector,
        )

    result_map = tensor_mapping(result)

    assert calls == [("memory.vector_residency", "cpu_pinned")]
    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_executes_gpu_vector_residency() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for memory.vector_residency=gpu")

    params = {"w": torch.tensor([2.0], dtype=torch.float64, device="cuda")}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "vector-residency-gpu",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.vector_residency": "gpu",
        },
        params,
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert result_map["w"].device.type == "cuda"
    assert torch.equal(result_map["w"].cpu(), torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_executes_metric_factor_residency_axis() -> None:
    calls = []
    original = runtime_module._residency_tensor
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}

    def recording_residency(
        tensor: torch.Tensor,
        residency: object,
        key: str,
    ) -> torch.Tensor:
        calls.append((key, residency, tuple(tensor.shape)))

        return original(tensor, residency, key)

    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runtime_module, "_residency_tensor", recording_residency)
        result = run_passed_candidate(
            factory,
            "metric",
            "factor-residency",
            {
                **metric_settings("factorized_multiply"),
                "memory.factor_residency": "cpu_staged",
            },
            {"low_rank_factors": factors},
            vector,
        )

    result_tensor = tree_leaves(result)[0]

    assert calls == [
        ("memory.factor_residency", "cpu_staged", (2, 1)),
        ("memory.factor_residency", "cpu_staged", (2,)),
    ]
    assert torch.allclose(
        result_tensor,
        torch.tensor([-0.25, -6.25], dtype=torch.float64),
    )


def test_standard_runtime_rejects_memory_mapped_vector_residency_without_binding() -> (
    None
):
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(vp.MaterializationError, match="memory-mapped tensor metadata"):
        factory(
            passed_candidate(
                "hvp",
                "vector-mmap",
                {
                    "hvp.path": "reverse_over_reverse",
                    "memory.vector_residency": "mmap_cpu",
                },
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_executes_custom_memory_mapped_vector_residency() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    calls = []

    def mmap_residency(tensor: torch.Tensor, key: str) -> torch.Tensor:
        calls.append((key, tuple(tensor.shape)))

        return tensor.detach().clone()

    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        mmap_residency=mmap_residency,
    )
    result = run_passed_candidate(
        factory,
        "hvp",
        "vector-mmap",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.vector_residency": "mmap_cpu",
        },
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert calls == [("memory.vector_residency", (1,))]
    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_rejects_memory_mapped_factor_residency_without_binding() -> (
    None
):
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = passed_candidate(
        "metric",
        "factor-mmap",
        {
            **metric_settings("factorized_multiply"),
            "memory.factor_residency": "mmap_cpu",
        },
    )

    with pytest.raises(vp.MaterializationError, match="memory-mapped tensor metadata"):
        factory(candidate, {"low_rank_factors": factors}, vector)


def test_standard_runtime_accepts_fresh_output_buffers() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "fresh-output",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.output_buffers": "fresh_allocation",
        },
        params,
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_executes_preallocated_output_buffers() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    first, second = repeated_standard_operation_maps(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params,
        {"scale": 1.0},
        vector,
        {
            "hvp.path": "reverse_over_reverse",
            "memory.output_buffers": "preallocated",
        },
        "preallocated-output",
        scalar_objectives={"loss": quadratic_scalar},
    )

    assert torch.equal(first["w"], torch.tensor([6.0], dtype=torch.float64))
    assert torch.equal(second["w"], torch.tensor([6.0], dtype=torch.float64))
    assert first["w"].data_ptr() == second["w"].data_ptr()

    with pytest.raises(vp.MaterializationError, match="preallocated output tree"):
        runtime_module._runtime_output_to_buffer(
            {"w": torch.ones(2, dtype=torch.float64)},
            {"w": torch.empty(1, dtype=torch.float64)},
        )


def test_preallocated_output_buffers_execute_metric_multiply() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    first, second = repeated_standard_operation_maps(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params,
        {"metric_matrix": DenseMetricData.matrix},
        vector,
        {
            **metric_settings("dense_matmul"),
            "memory.output_buffers": "preallocated",
        },
        "preallocated-metric",
    )
    expected = DenseMetricData.matrix @ vector["w"]

    assert torch.equal(first["w"], expected)
    assert torch.equal(second["w"], expected)
    assert first["w"].data_ptr() == second["w"].data_ptr()


def test_preallocated_output_buffers_execute_inverse_metric_solve() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    first, second = repeated_standard_operation_maps(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.25,
        ),
        params,
        {"metric_matrix": DenseMetricData.matrix},
        vector,
        {
            **inverse_metric_settings("dense_solve"),
            "memory.output_buffers": "preallocated",
        },
        "preallocated-inverse",
    )
    expected = torch.linalg.solve(
        DenseMetricData.matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )

    assert torch.equal(first["w"], expected)
    assert torch.equal(second["w"], expected)
    assert first["w"].data_ptr() == second["w"].data_ptr()


def test_preallocated_output_buffers_execute_jvp_function_output() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    first, second = repeated_standard_operation_maps(
        ops.jvp("jvp", "function", aggregation="none"),
        params,
        {"scale": 1.0},
        vector,
        {
            "jvp.path": "torch_func_jvp",
            "memory.output_buffers": "preallocated",
            **torch_func_settings(requires_forward_ad=True),
        },
        "preallocated-output",
        function_objectives={"function": square_function},
    )

    assert torch.equal(first["y"], torch.tensor([12.0], dtype=torch.float64))
    assert torch.equal(second["y"], torch.tensor([12.0], dtype=torch.float64))
    assert first["y"].data_ptr() == second["y"].data_ptr()


def test_standard_runtime_accepts_retained_memory_outputs() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "retain-memory-outputs",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.primal_outputs": "retain",
            "memory.jvp_outputs": "retain",
            "memory.output_cotangents": "retain",
        },
        params,
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


def test_standard_runtime_recomputes_hvp_primal_outputs() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([[3.0], [4.0]], dtype=torch.float64)}
    calls = {"scalar": 0}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        calls["scalar"] += 1

        return quadratic_scalar(params, buffers, batch, context)

    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        passed_candidate(
            "hvp",
            "recompute-primal-memory",
            {
                "hvp.path": "reverse_over_reverse",
                "hvp.primal_reuse": "recompute_primal",
                "memory.primal_outputs": "recompute",
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
        ),
        {"scale": 1.0},
        vector,
    )()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"] * 2.0)
    assert calls == {"scalar": 2}


def test_standard_runtime_recomputes_ggn_memory_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([0.5, -0.25], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -2.0], dtype=torch.float64)}
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    calls = {"function": 0, "cotangent": 0}
    original_loss_product = runtime_module._ggn_loss_hessian_product

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        calls["function"] += 1
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] ** 2 + params["w"][1],
            params["w"][0] - params["w"][1] ** 2,
        ))

    def recording_loss_product(
        execution: Any,
        output: vpx.TensorTree,
        output_jvp: vpx.TensorTree,
    ) -> vpx.TensorTree:
        calls["cotangent"] += 1

        return original_loss_product(execution, output, output_jvp)

    monkeypatch.setattr(
        runtime_module,
        "_ggn_loss_hessian_product",
        recording_loss_product,
    )
    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        passed_candidate(
            "ggn",
            "memory-recompute",
            {
                **ggn_reuse_settings(
                    "recompute_jvp",
                    "recompute_output_cotangent",
                ),
                "memory.jvp_outputs": "recompute",
                "memory.output_cotangents": "recompute",
            },
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()
    jacobian = torch.tensor(
        [[1.0, 1.0], [1.0, 0.5]],
        dtype=torch.float64,
    )
    expected = jacobian.T @ (loss_hessian @ (jacobian @ vector["w"]))

    torch.testing.assert_close(tree_leaves(result)[0], expected)
    assert calls == {"function": 3, "cotangent": 2}


@pytest.mark.parametrize(
    "setting_key",
    [
        "memory.primal_outputs",
        "memory.jvp_outputs",
        "memory.output_cotangents",
    ],
)
def test_standard_runtime_rejects_memory_recompute_without_lowering(
    setting_key: str,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(
        vp.MaterializationError,
        match="matching package-owned recompute settings",
    ):
        factory(
            passed_candidate(
                "hvp",
                "recompute-memory-output",
                {"hvp.path": "reverse_over_reverse", setting_key: "recompute"},
            ),
            {"scale": 1.0},
            vector,
        )


@pytest.mark.parametrize(
    "residency",
    ["gpu", "cpu_staged", "cpu_pinned"],
)
def test_standard_runtime_rejects_intermediate_residency_without_boundaries(
    residency: str,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(vp.MaterializationError, match="intermediate boundaries"):
        factory(
            passed_candidate(
                "hvp",
                "intermediate-residency",
                {
                    "hvp.path": "reverse_over_reverse",
                    "memory.intermediate_residency": residency,
                },
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_accepts_model_default_fusion_settings() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "model-default-fusion",
        {
            "hvp.path": "reverse_over_reverse",
            "fusion.norm": "model_default",
            "fusion.mlp": "model_default",
            "fusion.rope": "model_default",
            "fusion.logits": "model_default",
            "fusion.loss": "model_default",
        },
        params,
        {"scale": 1.0},
        vector,
    )
    result_map = tensor_mapping(result)

    assert torch.equal(result_map["w"], torch.tensor([6.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("setting_key", "setting_value"),
    [
        ("fusion.norm", "fused_rmsnorm"),
        ("fusion.norm", "fused_layernorm"),
        ("fusion.mlp", "fused_mlp"),
        ("fusion.rope", "fused_rope"),
        ("fusion.logits", "fused_logits_projection"),
    ],
)
def test_standard_runtime_rejects_fused_rows_without_registered_implementation(
    setting_key: str,
    setting_value: str,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = quadratic_hvp_factory(params)

    with pytest.raises(vp.MaterializationError, match="registered fused"):
        factory(
            passed_candidate(
                "hvp",
                "fused-row",
                {"hvp.path": "reverse_over_reverse", setting_key: setting_value},
            ),
            {"scale": 1.0},
            vector,
        )


@pytest.mark.parametrize(
    ("setting_key", "setting_value"),
    [
        ("fusion.norm", "fused_rmsnorm"),
        ("fusion.norm", "fused_layernorm"),
        ("fusion.mlp", "fused_mlp"),
        ("fusion.rope", "fused_rope"),
        ("fusion.logits", "fused_logits_projection"),
    ],
)
def test_standard_runtime_executes_fused_row_with_registered_rewriter(
    setting_key: str,
    setting_value: str,
) -> None:
    events = []

    class FusibleModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
            self.fused = False

        def forward(self, scale: torch.Tensor) -> torch.Tensor:
            events.append("fused" if self.fused else "default")

            return (self.w * scale).sum()

    def fusion_rewriter(
        module: torch.nn.Module,
        candidate: vpx.Candidate,
    ) -> torch.nn.Module:
        events.append(candidate.settings[setting_key])
        assert isinstance(module, FusibleModule)
        fused = FusibleModule()
        fused.fused = True

        return fused

    module = FusibleModule()
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers={},
        module=module,
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        fusion_rewriter=fusion_rewriter,
    )
    result = run_passed_candidate(
        factory,
        "gradient",
        "fused-row",
        {
            **gradient_settings(),
            **stateful_module_call_settings(),
            setting_key: setting_value,
        },
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )
    result_map = tensor_mapping(result)

    assert events == [setting_value, "fused"]
    torch.testing.assert_close(
        result_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    (
        "operator",
        "family",
        "settings",
        "batch",
        "vector",
        "scalar_objectives",
        "function_objectives",
        "expected",
    ),
    [
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            "hvp",
            {
                **hvp_settings("reverse_over_reverse"),
                "fusion.logits": "fused_logits_projection",
            },
            {"scale": torch.tensor([2.0], dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            {"loss": quadratic_scalar},
            None,
            torch.tensor([12.0], dtype=torch.float64),
        ),
        (
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            "ggn",
            {
                **ggn_dense_kernel_settings(),
                "fusion.logits": "fused_logits_projection",
            },
            {
                "scale": 1.0,
                "loss_hessian": torch.tensor([[5.0]], dtype=torch.float64),
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            None,
            {"model_output": square_function},
            torch.tensor([240.0], dtype=torch.float64),
        ),
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            {
                **fisher_settings("materialize_score_gradients"),
                "fusion.logits": "fused_logits_projection",
            },
            {
                "score_gradients": torch.tensor([[2.0], [4.0]], dtype=torch.float64),
                "normalization": 2.0,
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            None,
            None,
            torch.tensor([30.0], dtype=torch.float64),
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            {
                **sampled_fisher_settings("materialize_score_gradients"),
                "fusion.logits": "fused_logits_projection",
            },
            {
                "sampled_score_gradients": torch.tensor(
                    [[2.0], [4.0]], dtype=torch.float64
                ),
                "num_examples": 1,
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            None,
            None,
            torch.tensor([30.0], dtype=torch.float64),
        ),
        (
            empirical_fisher_sum("empirical", "scores"),
            "empirical",
            {
                **empirical_dense_settings(),
                "fusion.logits": "fused_logits_projection",
            },
            {
                "per_example_gradients": torch.tensor(
                    [[2.0], [4.0]], dtype=torch.float64
                ),
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            None,
            None,
            torch.tensor([60.0], dtype=torch.float64),
        ),
    ],
)
def test_standard_runtime_executes_fused_rows_for_higher_order_families(
    operator: vpx.OperatorSpec,
    family: str,
    settings: Mapping[str, object],
    batch: Mapping[str, object],
    vector: vpx.TensorTree,
    scalar_objectives: Mapping[str, Callable[..., torch.Tensor]] | None,
    function_objectives: Mapping[str, Callable[..., vpx.TensorTree]] | None,
    expected: torch.Tensor,
) -> None:
    events = []

    def fusion_rewriter(
        module: torch.nn.Module,
        candidate: vpx.Candidate,
    ) -> torch.nn.Module:
        events.append((candidate.family, candidate.settings["fusion.logits"]))

        return module

    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=OneParameterModule(),
        fusion_rewriter=fusion_rewriter,
    )
    result = run_passed_candidate(
        factory,
        family,
        "fused-higher-order",
        settings,
        batch,
        vector,
    )

    assert events == [(family, "fused_logits_projection")]
    torch.testing.assert_close(tree_leaves(result)[0], expected)


def test_standard_runtime_rejects_fused_loss_without_loss_identity() -> None:
    def fusion_rewriter(
        module: torch.nn.Module,
        candidate: vpx.Candidate,
    ) -> torch.nn.Module:
        _ = candidate

        return module

    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        module=OneParameterModule(),
        fusion_rewriter=fusion_rewriter,
    )

    with pytest.raises(vp.MaterializationError, match="typed softmax_cross_entropy"):
        run_passed_candidate(
            factory,
            "hvp",
            "fused-loss",
            {**hvp_settings("reverse_over_reverse"), "fusion.loss": "fused_ce"},
            {"scale": torch.tensor([2.0], dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_standard_runtime_rejects_fused_loss_kind_mismatch() -> None:
    def fusion_rewriter(
        module: torch.nn.Module,
        candidate: vpx.Candidate,
    ) -> torch.nn.Module:
        _ = candidate

        return module

    operator = dataclasses.replace(
        ops.hvp("hvp", "loss", aggregation="sum"),
        semantics={
            "loss": {
                "kind": "softmax_cross_entropy",
                "identity": {
                    "reduction": "token_mean",
                    "denominator": "num_tokens",
                },
            }
        },
    )
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        module=OneParameterModule(),
        fusion_rewriter=fusion_rewriter,
    )

    with pytest.raises(vp.MaterializationError, match="typed kl"):
        run_passed_candidate(
            factory,
            "hvp",
            "fused-loss",
            {**hvp_settings("reverse_over_reverse"), "fusion.loss": "fused_kl"},
            {"scale": torch.tensor([2.0], dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_standard_runtime_accepts_fused_loss_with_normalization_identity() -> None:
    events = []

    def fusion_rewriter(
        module: torch.nn.Module,
        candidate: vpx.Candidate,
    ) -> torch.nn.Module:
        events.append(candidate.settings["fusion.loss"])

        return module

    operator = dataclasses.replace(
        ops.hvp("hvp", "loss", aggregation="sum"),
        semantics={
            "loss": {
                "kind": "softmax_cross_entropy",
                "identity": {
                    "reduction": "token_mean",
                    "denominator": "num_tokens",
                },
            }
        },
    )
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        module=OneParameterModule(),
        fusion_rewriter=fusion_rewriter,
    )
    result = run_passed_candidate(
        factory,
        "hvp",
        "fused-loss",
        {**hvp_settings("reverse_over_reverse"), "fusion.loss": "fused_ce"},
        {"scale": torch.tensor([2.0], dtype=torch.float64)},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert events == ["fused_ce"]
    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([12.0], dtype=torch.float64),
    )


def test_standard_runtime_executes_layout_contiguity_axis() -> None:
    params = {"w": torch.arange(4, dtype=torch.float64).reshape(2, 2).T}
    buffers = {"b": torch.arange(4, dtype=torch.float64).reshape(2, 2).T}
    batch = {"floating": torch.arange(4, dtype=torch.float64).reshape(2, 2).T}
    vector = {"w": torch.ones((2, 2), dtype=torch.float64).T}
    observed = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert context.family == "gradient"
        observed.append((
            params["w"].is_contiguous(),
            buffers["b"].is_contiguous(),
            batch["floating"].is_contiguous(),
        ))

        return (params["w"] * batch["floating"]).sum() + buffers["b"].sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )

    for value in ("preserve_existing_strides", "contiguous"):
        run_passed_candidate(
            factory,
            "gradient",
            value,
            {
                **gradient_settings(),
                "layout.contiguity": value,
                "layout.flatten_order": "canonical_parameter_order",
            },
            batch,
            vector,
        )

    assert observed == [(False, False, False), (True, True, True)]

    with pytest.raises(vp.MaterializationError, match=r"layout\.flatten_order"):
        factory(
            passed_candidate(
                "gradient",
                "bad-flatten-order",
                {**gradient_settings(), "layout.flatten_order": "by_name"},
            ),
            batch,
            vector,
        )


def test_standard_runtime_executes_layout_output_flat_contiguous() -> None:
    params = {
        "a": torch.tensor([2.0], dtype=torch.float64),
        "b": torch.tensor([3.0, 4.0], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["scale"]
        assert context.family == "gradient"

        return 0.5 * (params["a"].pow(2).sum() + params["b"].pow(2).sum())

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    candidate = passed_candidate(
        "gradient",
        "flat-output",
        {
            **gradient_settings(),
            "layout.output": "flat_contiguous",
            "layout.aliasing": "preserve_tied_weight_aliases",
            "layout.parametrizations": "preserve_active_parametrizations",
        },
    )
    output = factory(candidate, {"scale": 1.0}, vector)()
    check = vpx.standard_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "directional_abs_diff": 1e-3,
            "directional_rel_diff": 1e-3,
        },
        scalar_objectives={"loss": scalar},
    )
    reference_result = check(candidate, {"scale": 1.0}, vector)

    assert isinstance(output, torch.Tensor)
    assert output.is_contiguous()
    torch.testing.assert_close(
        output,
        torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
    )
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("case_id", "group_kind", "settings", "same_pairs", "different_pairs"),
    [
        (
            "per-layer",
            "layer",
            {
                "layout.params": "per_layer_flat",
                "layout.output": "per_layer_flat",
            },
            (("a", "b"),),
            (("a", "c"),),
        ),
        (
            "per-layer-output",
            "layer",
            {"layout.output": "per_layer_flat"},
            (("a", "b"),),
            (("a", "c"),),
        ),
        (
            "per-block",
            "block",
            {
                "layout.params": "per_block_flat",
                "layout.output": "per_block_flat",
            },
            (("b", "c"),),
            (("a", "b"),),
        ),
        (
            "per-block-output",
            "block",
            {"layout.output": "per_block_flat"},
            (("b", "c"),),
            (("a", "b"),),
        ),
    ],
)
def test_standard_runtime_executes_layout_output_alias_groups(
    case_id: str,
    group_kind: str,
    settings: dict[str, object],
    same_pairs: tuple[tuple[str, str], ...],
    different_pairs: tuple[tuple[str, str], ...],
) -> None:
    params = {
        "a": torch.tensor([2.0], dtype=torch.float64),
        "b": torch.tensor([3.0, 4.0], dtype=torch.float64),
        "c": torch.tensor([5.0], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "c": torch.tensor([1.0], dtype=torch.float64),
    }

    if group_kind == "layer":
        parameter_surface = vpx.ParameterSurface(
            names=("a", "b", "c"),
            shapes=((1,), (2,), (1,)),
            trainable=(True, True, True),
            layer_groups=(("a", "b"), ("c",)),
        )
    else:
        parameter_surface = vpx.ParameterSurface(
            names=("a", "b", "c"),
            shapes=((1,), (2,), (1,)),
            trainable=(True, True, True),
            block_groups=(("a",), ("b", "c")),
        )

    def storage_id(tensor: torch.Tensor) -> object:
        return tensor.untyped_storage()._cdata

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "gradient"

        return 0.5 * (
            params["a"].pow(2).sum()
            + params["b"].pow(2).sum()
            + params["c"].pow(2).sum()
        )

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        parameter_surface=parameter_surface,
        scalar_objectives={"loss": scalar},
    )
    result = run_passed_candidate(
        factory,
        "gradient",
        case_id,
        {**gradient_settings(), **settings},
        {},
        vector,
    )
    result_map = tensor_mapping(result)

    for left, right in same_pairs:
        assert storage_id(result_map[left]) == storage_id(result_map[right])

    for left, right in different_pairs:
        assert storage_id(result_map[left]) != storage_id(result_map[right])

    torch.testing.assert_close(result_map["a"], params["a"])
    torch.testing.assert_close(result_map["b"], params["b"])
    torch.testing.assert_close(result_map["c"], params["c"])


@pytest.mark.parametrize(
    ("case_id", "group_kind", "layout_value"),
    [
        ("per-layer-vector", "layer", "per_layer_flat"),
        ("per-block-vector", "block", "per_block_flat"),
    ],
)
def test_standard_runtime_executes_grouped_flat_vector_layout(
    case_id: str,
    group_kind: str,
    layout_value: str,
) -> None:
    params = {
        "a": torch.tensor([0.0], dtype=torch.float64),
        "b": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "c": torch.tensor([0.0], dtype=torch.float64),
    }
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    if group_kind == "layer":
        parameter_surface = vpx.ParameterSurface(
            names=("a", "b", "c"),
            shapes=((1,), (2,), (1,)),
            trainable=(True, True, True),
            layer_groups=(("a", "b"), ("c",)),
        )
    else:
        parameter_surface = vpx.ParameterSurface(
            names=("a", "b", "c"),
            shapes=((1,), (2,), (1,)),
            trainable=(True, True, True),
            block_groups=(("a",), ("b", "c")),
        )

    factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
        parameter_surface=parameter_surface,
    )
    result = run_passed_candidate(
        factory,
        "metric",
        case_id,
        {
            **metric_settings("dense_matmul"),
            "layout.vector": layout_value,
        },
        {"metric_matrix": torch.eye(4, dtype=torch.float64)},
        vector,
    )
    result_map = tensor_mapping(result)

    torch.testing.assert_close(
        result_map["a"], torch.tensor([1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result_map["b"],
        torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result_map["c"], torch.tensor([4.0], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("layout.params", "per_layer_flat"),
        ("layout.vector", "per_layer_flat"),
        ("layout.params", "per_block_flat"),
        ("layout.vector", "per_block_flat"),
        ("layout.params", "per_shard"),
        ("layout.vector", "per_shard"),
        ("layout.params", "dtensor"),
        ("layout.vector", "dtensor"),
    ],
)
def test_standard_runtime_rejects_non_tree_input_layout_without_reconstruction(
    key: str,
    value: str,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    if value == "per_layer_flat":
        match = "declared layer_groups"
    elif value == "per_block_flat":
        match = "declared block_groups"
    else:
        match = "tree reconstruction support"

    with pytest.raises(vp.MaterializationError, match=match):
        factory(
            passed_candidate(
                "gradient",
                "non-tree-input-layout",
                {**gradient_settings(), key: value},
            ),
            {"scale": 1.0},
            vector,
        )


@pytest.mark.parametrize("value", ["per_shard", "dtensor"])
def test_standard_runtime_rejects_distributed_output_layout_without_adapter(
    value: str,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(
        vp.MaterializationError, match=r"layout[.]output is unsupported"
    ):
        factory(
            passed_candidate(
                "gradient",
                "distributed-output-layout",
                {**gradient_settings(), "layout.output": value},
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_executes_layout_params_flat_contiguous() -> None:
    params = {
        "a": torch.tensor([2.0], dtype=torch.float64),
        "b": torch.tensor([3.0, 4.0], dtype=torch.float64),
    }
    vector = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    settings = {**gradient_settings(), "layout.params": "flat_contiguous"}
    runtime_params = runtime_module._runtime_params(params, settings, None)
    observed = []

    assert (
        runtime_params["a"].untyped_storage().data_ptr()
        == runtime_params["b"].untyped_storage().data_ptr()
    )

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "gradient"
        observed.append((
            params["a"].is_contiguous(),
            params["b"].is_contiguous(),
        ))

        return 0.5 * (params["a"].pow(2).sum() + params["b"].pow(2).sum())

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "flat-params",
            settings,
            admission_status="passed",
        ),
        {},
        vector,
    )()

    assert observed == [(True, True)]
    torch.testing.assert_close(tree_leaves(result)[0], params["a"])
    torch.testing.assert_close(tree_leaves(result)[1], params["b"])


def test_standard_runtime_preserves_tied_parameter_aliases_during_dtype_cast() -> None:
    shared = torch.tensor([2.0], dtype=torch.float64)
    params = {"a": shared, "b": shared}
    settings = {
        **gradient_settings(),
        "dtype.parameter_storage": "fp32",
        "call.tied_weights": "preserve_alias_groups",
    }

    runtime_params = runtime_module._runtime_params(params, settings, None)

    assert runtime_params["a"] is runtime_params["b"]
    assert runtime_params["a"].dtype == torch.float32


def test_standard_runtime_rejects_flat_params_when_tied_aliases_must_preserve() -> None:
    shared = torch.tensor([2.0], dtype=torch.float64)
    params = {"a": shared, "b": shared}

    with pytest.raises(vp.MaterializationError, match="tied parameter aliases"):
        runtime_module._runtime_params(
            params,
            {
                **gradient_settings(),
                "layout.params": "flat_contiguous",
                "layout.aliasing": "preserve_tied_weight_aliases",
            },
            None,
        )


def test_standard_runtime_rejects_alias_preservation_on_deduplicated_surface() -> None:
    params = {"a": torch.tensor([2.0], dtype=torch.float64)}
    parameter_surface = vpx.ParameterSurface(
        names=("a",),
        shapes=((1,),),
        trainable=(True,),
        tied_weights_policy="deduplicate",
    )

    with pytest.raises(vp.MaterializationError, match="preserved parameter surface"):
        runtime_module._runtime_params(
            params,
            {
                **gradient_settings(),
                "call.tied_weights": "preserve_alias_groups",
            },
            parameter_surface,
        )


@pytest.mark.parametrize(
    "vector",
    [
        {
            "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
            "a": torch.tensor([1.0], dtype=torch.float64),
        },
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    ],
)
def test_standard_runtime_executes_layout_vector_flat_contiguous(
    vector: vpx.TensorTree,
) -> None:
    params = {
        "a": torch.tensor([0.0], dtype=torch.float64),
        "b": torch.tensor([0.0, 0.0], dtype=torch.float64),
    }
    matrix = torch.diag(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
    factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
    )
    result = factory(
        vpx.Candidate(
            "metric",
            "flat-vector",
            {
                **metric_settings("dense_matmul"),
                "layout.vector": "flat_contiguous",
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )()

    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([10.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tree_leaves(result)[1],
        torch.tensor([40.0, 90.0], dtype=torch.float64),
    )


def test_standard_runtime_executes_call_grad_mode_axis() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "gradient"
        observed.append(torch.is_grad_enabled())

        return params["w"].pow(2).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )

    with torch.no_grad():
        result = factory(
            vpx.Candidate(
                "gradient",
                "grad-enabled",
                {**gradient_settings(), "call.grad_mode": "grad_enabled"},
                admission_status="passed",
            ),
            {},
            vector,
        )()

    assert observed == [True]
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([4.0], dtype=torch.float64),
    )


def test_standard_runtime_executes_explicit_functional_call_settings() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([3.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert batch == {}
        observed.append((
            params["w"].detach().clone(),
            buffers["b"].detach().clone(),
            dict(context.settings),
        ))

        return (params["w"] * buffers["b"]).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )
    settings = {
        **gradient_settings(),
        "call.path": "functional_call",
        "call.params": "explicit_params",
        "call.buffers": "explicit_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "forbidden",
        "call.grad_mode": "grad_enabled",
        "call.return_type": "raw_tensor_tree",
    }
    result = factory(
        vpx.Candidate(
            "gradient",
            "functional-call",
            settings,
            admission_status="passed",
        ),
        {},
        vector,
    )()

    assert len(observed) == 1
    observed_params, observed_buffers, observed_settings = observed[0]
    assert torch.equal(observed_params, params["w"])
    assert torch.equal(observed_buffers, buffers["b"])
    assert observed_settings == settings
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([3.0], dtype=torch.float64),
    )


def test_standard_runtime_rejects_non_tree_raw_function_output() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def tensor_function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> Any:
        assert isinstance(params["w"], torch.Tensor)
        assert buffers == {}
        assert batch == {}
        assert context.family == "jvp"

        return object()

    factory = vpx.standard_operation_factory(
        ops.jvp("jvp", "function", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"function": tensor_function},
    )

    with pytest.raises(vp.MaterializationError, match="raw tensor tree"):
        factory(
            vpx.Candidate(
                "jvp",
                "non-tree-output",
                {
                    **jvp_settings("torch_func_jvp"),
                    "call.return_type": "raw_tensor_tree",
                    **torch_func_settings(requires_forward_ad=True),
                },
                admission_status="passed",
            ),
            {},
            vector,
        )()


def test_standard_runtime_rejects_forbidden_functional_buffer_mutation() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([3.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert batch == {}
        assert context.family == "gradient"
        buffers["b"].add_(1.0)

        return (params["w"] * buffers["b"]).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )

    with pytest.raises(vp.MaterializationError, match="changed buffer value: b"):
        factory(
            vpx.Candidate(
                "gradient",
                "forbidden-buffer-mutation",
                {
                    **gradient_settings(),
                    "call.path": "functional_call",
                    "call.params": "explicit_params",
                    "call.buffers": "explicit_buffers",
                    "call.buffer_mutation": "forbidden",
                },
                admission_status="passed",
            ),
            {},
            vector,
        )()


def test_standard_runtime_restores_declared_functional_buffer_mutation() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([3.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": mutating_buffer_scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "declared-restored-buffer-mutation",
            {
                **gradient_settings(),
                **declared_restored_functional_call_settings(),
            },
            admission_status="passed",
        ),
        {},
        vector,
    )()

    assert torch.equal(buffers["b"], torch.tensor([3.0], dtype=torch.float64))
    assert torch.equal(tree_leaves(result)[0], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_restores_declared_buffer_mutation_after_error() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([3.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": failing_mutating_buffer_scalar},
    )

    with pytest.raises(RuntimeError, match="declared mutation failure"):
        factory(
            vpx.Candidate(
                "gradient",
                "declared-restored-buffer-mutation-error",
                {
                    **gradient_settings(),
                    **declared_restored_functional_call_settings(),
                },
                admission_status="passed",
            ),
            {},
            vector,
        )()

    assert torch.equal(buffers["b"], torch.tensor([3.0], dtype=torch.float64))


def stateful_module_call_settings() -> dict[str, object]:
    return {
        "call.path": "stateful_module",
        "call.params": "module_params",
        "call.buffers": "module_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "forbidden",
        "call.return_type": "raw_tensor_tree",
    }


def test_standard_runtime_executes_stateful_module_gradient() -> None:
    module = StatefulScalarModule()
    original_parameter = module.w
    original_buffer = module.b
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=dict(module.named_buffers()),
        module=module,
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "stateful-module-gradient",
            {**gradient_settings(), **stateful_module_call_settings()},
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()
    result_map = tensor_mapping(result)

    assert module.w is original_parameter
    assert module.b is original_buffer
    torch.testing.assert_close(
        result_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_standard_runtime_restores_declared_stateful_module_buffer() -> None:
    module = MutatingStatefulScalarModule()
    buffers = dict(module.named_buffers())
    settings = {
        **declared_restored_functional_call_settings(mutated_buffer_keys=("b",)),
        "call.path": "stateful_module",
        "call.params": "module_params",
        "call.buffers": "module_buffers",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=buffers,
        module=module,
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "stateful-module-restored",
            {**gradient_settings(), **settings},
            admission_status="passed",
        ),
        {"scale": torch.tensor([2.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()
    result_map = tensor_mapping(result)

    torch.testing.assert_close(
        result_map["w"], torch.tensor([5.0], dtype=torch.float64)
    )
    torch.testing.assert_close(module.b, torch.tensor([3.0], dtype=torch.float64))
    torch.testing.assert_close(buffers["b"], torch.tensor([3.0], dtype=torch.float64))


def test_standard_runtime_selects_stateful_module_output_fields_for_vjp() -> None:
    module = StatefulOutputModule()
    factory = vpx.standard_operation_factory(
        ops.vjp("vjp", "model_output", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=dict(module.named_buffers()),
        module=module,
        module_call=vpx.ModuleCallSpec(
            positional_batch_keys=("scale",),
            output_fields={"logits": ("logits",)},
        ),
    )
    result = factory(
        vpx.Candidate(
            "vjp",
            "stateful-module-vjp",
            {
                "vjp.path": "autograd_grad_outputs",
                **stateful_module_call_settings(),
                "call.return_type": "model_output_object_with_declared_fields",
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor([3.0, 5.0], dtype=torch.float64)},
        {"logits": torch.tensor([7.0, 11.0], dtype=torch.float64)},
    )()
    result_map = tensor_mapping(result)

    torch.testing.assert_close(
        result_map["w"],
        torch.tensor([21.0, 55.0], dtype=torch.float64),
    )


def test_standard_runtime_stateful_module_requires_binding() -> None:
    module = StatefulScalarModule()
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=dict(module.named_buffers()),
    )

    with pytest.raises(vp.MaterializationError, match="requires a module"):
        factory(
            vpx.Candidate(
                "gradient",
                "stateful-module-missing-binding",
                {**gradient_settings(), **stateful_module_call_settings()},
                admission_status="passed",
            ),
            {"scale": torch.tensor([4.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_rejects_restore_mode_without_declared_mutation() -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={"b": torch.tensor([3.0], dtype=torch.float64)},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="mutates_state=True"):
        factory(
            vpx.Candidate(
                "gradient",
                "restore-without-mutates-state",
                {
                    **gradient_settings(),
                    **declared_restored_functional_call_settings(
                        mutates_state=False,
                        mutated_buffer_keys=(),
                    ),
                },
                admission_status="passed",
            ),
            {},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


@pytest.mark.parametrize(
    ("settings_override", "message"),
    [
        ({"call.path": "functional_call"}, "incomplete"),
        (
            {
                "call.path": "stateful_module",
                "call.params": "explicit_params",
                "call.buffers": "explicit_buffers",
            },
            "call.params=module_params",
        ),
        (
            {
                "call.path": "functional_call",
                "call.params": "module_params",
                "call.buffers": "explicit_buffers",
            },
            "call.params is unsupported",
        ),
        (
            {
                "call.path": "functional_call",
                "call.params": "explicit_params",
                "call.buffers": "module_buffers",
            },
            "call.buffers is unsupported",
        ),
        (
            {"call.buffer_mutation": "declared_and_restored"},
            "explicit functional-call inputs",
        ),
        (
            {"call.return_type": "model_output_object_with_declared_fields"},
            "call.return_type is unsupported",
        ),
    ],
)
def test_standard_runtime_rejects_call_rows_without_required_binding(
    settings_override: Mapping[str, object],
    message: str,
) -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match=message):
        factory(
            vpx.Candidate(
                "gradient",
                "call-row",
                {**gradient_settings(), **settings_override},
                admission_status="passed",
            ),
            {"scale": torch.tensor(1.0, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_accepts_direct_input_schedule_settings() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        observed.append(dict(context.settings))

        return params["w"].pow(2).sum() * batch["scale"]

    settings = {
        **gradient_settings(),
        "input.batch_layout": "dense_padded",
        "input.length_grouping": "none",
        "schedule.per_token": "loop",
        "schedule.gradient_accumulation": "single_step",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "direct-input-schedule",
            settings,
            admission_status="passed",
        ),
        {"scale": torch.tensor(3.0, dtype=torch.float64)},
        vector,
    )()

    assert observed == [settings]
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([12.0], dtype=torch.float64),
    )


def microbatch_accumulate_result(
    operator: Any,
    settings: Mapping[str, object],
    objective_kind: str,
    parameter_power: int,
) -> tuple[object, list[tuple[int, object]]]:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == operator.family
        calls.append((batch["x"].shape[0], batch["scale"]))

        return (params["w"][0].pow(parameter_power) * batch["x"]).sum() * batch["scale"]

    if objective_kind == "function":
        factory = vpx.standard_operation_factory(
            operator,
            params=params,
            buffers={},
            function_objectives={"model": objective},
        )
    elif objective_kind == "scalar":
        factory = vpx.standard_operation_factory(
            operator,
            params=params,
            buffers={},
            scalar_objectives={"loss": objective},
        )
    else:
        raise AssertionError(objective_kind)

    result = factory(
        vpx.Candidate(
            operator.family,
            "microbatch",
            settings,
            admission_status="passed",
        ),
        {
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
            "scale": 3.0,
        },
        vector,
    )()

    return result, calls


@pytest.mark.parametrize(
    (
        "operator",
        "settings",
        "objective_kind",
        "parameter_power",
        "result_shape",
        "expected",
    ),
    [
        (
            ops.gradient("gradient", "loss", aggregation="sum"),
            {
                **gradient_settings(),
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            "scalar",
            1,
            "tree",
            torch.tensor([45.0], dtype=torch.float64),
        ),
        (
            ops.jvp("jvp", "model", aggregation="sum"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            "function",
            1,
            "tensor",
            torch.tensor(45.0, dtype=torch.float64),
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {
                **hvp_settings("reverse_over_reverse"),
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            "scalar",
            2,
            "tree",
            torch.tensor([90.0], dtype=torch.float64),
        ),
    ],
)
def test_microbatch_accumulate_runs_declared_data_chunks(
    operator: Any,
    settings: Mapping[str, object],
    objective_kind: str,
    parameter_power: int,
    result_shape: str,
    expected: torch.Tensor,
) -> None:
    result, calls = microbatch_accumulate_result(
        operator,
        settings,
        objective_kind,
        parameter_power,
    )
    actual = {
        "tensor": lambda value: value,
        "tree": lambda value: tree_leaves(value)[0],
    }[result_shape](result)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
    assert calls == [(2, 3.0), (2, 3.0), (1, 3.0)]


@pytest.mark.parametrize(
    ("settings_override", "message"),
    [
        ({"input.batch_layout": "packed_with_inverse_permutation"}, "input-layout"),
        ({"input.batch_layout": "variable_length"}, "input-layout"),
        ({"input.length_grouping": "exact_length_bucket"}, "input-order"),
        ({"schedule.per_token": "packed"}, "packed input binding"),
    ],
)
def test_standard_runtime_rejects_input_schedule_rows_without_binding(
    settings_override: Mapping[str, object],
    message: str,
) -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match=message):
        factory(
            vpx.Candidate(
                "gradient",
                "input-schedule",
                {**gradient_settings(), **settings_override},
                admission_status="passed",
            ),
            {"scale": torch.tensor(1.0, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_executes_packed_input_layout_binding() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        restored = batch["packed_x"].index_select(0, batch["inverse_permutation"])

        return params["w"][0] * restored.sum()

    def batch_layout(candidate: vpx.Candidate, batch: vpx.Batch) -> vpx.Batch:
        calls.append(candidate.settings["input.batch_layout"])
        permutation = torch.tensor([2, 0, 1])
        inverse_permutation = torch.tensor([1, 2, 0])

        return {
            "packed_x": batch["x"].index_select(0, permutation),
            "inverse_permutation": inverse_permutation,
        }

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
        batch_layout=batch_layout,
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "packed-input",
            {
                **gradient_settings(),
                "input.batch_layout": "packed_with_inverse_permutation",
                "input.length_grouping": "exact_length_bucket",
                "schedule.per_token": "packed",
            },
            admission_status="passed",
        ),
        {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert calls == ["packed_with_inverse_permutation"]
    torch.testing.assert_close(
        result_map["w"], torch.tensor([6.0], dtype=torch.float64)
    )


def test_standard_runtime_rejects_unlowered_chunk_axis() -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match=r"schedule[.]per_token"):
        factory(
            vpx.Candidate(
                "gradient",
                "chunk-row",
                {**gradient_settings(), "chunk.token_block_size": 2},
                admission_status="passed",
            ),
            {"scale": torch.tensor(1.0, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_executes_lm_head_chunker_binding() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return params["w"][0] * batch["logits"].sum()

    def lm_head_chunker(candidate: vpx.Candidate, batch: vpx.Batch) -> vpx.Batch:
        calls.append(candidate.settings["chunk.lm_head_weight_chunk_bytes"])
        hidden = batch["hidden"]
        weight = batch["lm_head_weight"]
        logits = hidden @ weight.T

        return {**batch, "logits": logits}

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
        lm_head_chunker=lm_head_chunker,
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "lm-head-chunk",
            {
                **gradient_settings(),
                "chunk.lm_head_weight_chunk_bytes": 16,
            },
            admission_status="passed",
        ),
        {
            "hidden": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            "lm_head_weight": torch.tensor(
                [[3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float64,
            ),
        },
        vector,
    )()
    result_map = tensor_mapping(result)

    assert calls == [16]
    torch.testing.assert_close(
        result_map["w"], torch.tensor([28.0], dtype=torch.float64)
    )


def test_standard_runtime_rejects_lm_head_chunking_without_binding() -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="LM-head weight binding"):
        factory(
            vpx.Candidate(
                "gradient",
                "lm-head-chunk",
                {
                    **gradient_settings(),
                    "chunk.lm_head_weight_chunk_bytes": 16,
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor(1.0, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def gradient_graph_schedule_result(
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
) -> tuple[list[str], vpx.TensorTree, vpx.TensorTree]:
    builds = []
    original_grad = torch.func.grad

    def recording_grad(function: Any) -> Any:
        builds.append("grad")

        return original_grad(function)

    monkeypatch.setattr(torch.func, "grad", recording_grad)
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            "graph-schedule",
            {
                "gradient.path": "torch_func_grad",
                **torch_func_settings(requires_forward_ad=False),
                "gradient.graph_schedule": schedule,
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor(1.0, dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    return builds, operation(), operation()


@pytest.mark.parametrize(
    ("schedule", "expected_builds"),
    [
        ("build_once", ["grad"]),
        ("rebuild_per_call", ["grad", "grad"]),
    ],
)
def test_gradient_graph_schedule_controls_gradient_builds(
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
    expected_builds: list[str],
) -> None:
    builds, first, second = gradient_graph_schedule_result(
        monkeypatch,
        schedule,
    )

    assert builds == expected_builds
    torch.testing.assert_close(
        tree_leaves(first)[0],
        torch.tensor([4.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tree_leaves(second)[0],
        torch.tensor([4.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("settings_override", "message"),
    [
        (
            {"schedule.gradient_accumulation": "microbatch_accumulate"},
            "batch.data_microbatch_size",
        ),
        (
            {"batch.data_microbatch_size": 2},
            "schedule.gradient_accumulation=microbatch_accumulate",
        ),
        (
            {
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 0,
            },
            "positive integer",
        ),
        (
            {
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
                **compile_settings(boundary="loss_closure"),
            },
            "loss_closure",
        ),
    ],
)
def test_gradient_microbatch_accumulate_rejects_invalid_settings(
    settings_override: Mapping[str, object],
    message: str,
) -> None:
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match=message):
        factory(
            vpx.Candidate(
                "gradient",
                "microbatch",
                {**gradient_settings(), **settings_override},
                admission_status="passed",
            ),
            {"scale": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_microbatch_accumulate_rejects_non_sum_operator() -> None:
    def model(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "jvp"

        return (params["w"][0] * batch["x"]).sum()

    factory = vpx.standard_operation_factory(
        ops.jvp("jvp", "model", aggregation="none"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"model": model},
    )
    operation = factory(
        vpx.Candidate(
            "jvp",
            "microbatch",
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            admission_status="passed",
        ),
        {"x": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    with pytest.raises(vp.MaterializationError, match="sum aggregation"):
        operation()


@pytest.mark.parametrize(
    ("settings_override", "message"),
    [
        ({"schedule.per_example": "loop"}, "incompatible"),
        (
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                "schedule.per_example": "vmap",
            },
            "incompatible",
        ),
        (
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="vmap_grad",
                ),
                "schedule.per_example": "manual_batch",
            },
            "incompatible",
        ),
    ],
)
def test_standard_runtime_rejects_incompatible_per_example_schedule(
    settings_override: Mapping[str, object],
    message: str,
) -> None:
    def scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"

        return params["w"] * batch["scale"]

    if "fisher.accumulation" in settings_override:
        operator = score_terms_fisher("fisher", "scores")
        settings = dict(settings_override)
        function_objectives = {"scores": scores}
    else:
        operator = ops.gradient("gradient", "loss", aggregation="sum")
        settings = {**gradient_settings(), **settings_override}
        function_objectives = {}

    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        function_objectives=function_objectives,
    )

    with pytest.raises(vp.MaterializationError, match=message):
        factory(
            vpx.Candidate(
                operator.family,
                "per-example",
                settings,
                admission_status="passed",
            ),
            {"scale": torch.tensor(1.0, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_diagonal_inverse_metric_factorized_solve_matches_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    diagonal = {"w": torch.tensor([4.0, 5.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "diagonal",
        aggregation="sum",
        representation=diagonal_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 2.0,
        },
    )
    batch = {"metric_diagonal": diagonal}
    candidate = vpx.Candidate(
        "inverse",
        "factorized",
        inverse_metric_settings("factorized_solve"),
        admission_status="passed",
    )
    expected = vector["w"] / (diagonal["w"] + 0.25)
    output = factory(candidate, batch, vector)()
    reference_result = check(candidate, batch, vector)

    assert torch.allclose(tree_leaves(output)[0], expected)
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert reference_result.measurements["damping_min"] == pytest.approx(0.25)


def test_block_metric_paths_match_dense_reference() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([0.25], dtype=torch.float64),
        "b": torch.tensor([-0.75, 0.5], dtype=torch.float64),
    }
    operator = ops.metric(
        "metric",
        "blocks",
        aggregation="sum",
        representation=block_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    batch = {"metric_blocks": blocks}
    expected = torch.block_diag(*blocks) @ flatten_tree(vector)

    for path in ("blockwise_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(flatten_tree(output), expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_block_metric_schedule_must_match_representation() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([0.25], dtype=torch.float64),
        "b": torch.tensor([-0.75, 0.5], dtype=torch.float64),
    }
    operator = ops.metric(
        "metric",
        "blocks",
        aggregation="sum",
        representation={"kind": "block_diagonal", "block_schedule": "custom_blocks"},
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    output = factory(
        vpx.Candidate(
            "metric",
            "custom-blocks",
            {
                **metric_settings("blockwise_multiply"),
                "metric.block_schedule": "custom_blocks",
            },
            admission_status="passed",
        ),
        {"metric_blocks": blocks},
        vector,
    )()

    assert torch.allclose(
        flatten_tree(output), torch.block_diag(*blocks) @ flatten_tree(vector)
    )

    with pytest.raises(
        vp.MaterializationError,
        match=r"metric[.]block_schedule must match representation[.]block_schedule",
    ):
        factory(
            vpx.Candidate(
                "metric",
                "layer-blocks",
                {
                    **metric_settings("blockwise_multiply"),
                    "metric.block_schedule": "layer_blocks",
                },
                admission_status="passed",
            ),
            {"metric_blocks": blocks},
            vector,
        )


def test_block_inverse_metric_solve_matches_dense_reference() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([0.25], dtype=torch.float64),
        "b": torch.tensor([-0.75, 0.5], dtype=torch.float64),
    }
    operator = ops.inverse_metric(
        "inverse",
        "blocks",
        aggregation="sum",
        representation=block_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 5.0,
        },
    )
    batch = {"metric_blocks": blocks}
    dense_matrix = torch.block_diag(*blocks)
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(3, dtype=torch.float64),
        flatten_tree(vector),
    )
    candidate = vpx.Candidate(
        "inverse",
        "blockwise",
        inverse_metric_settings("blockwise_solve"),
        admission_status="passed",
    )
    output = factory(candidate, batch, vector)()
    reference_result = check(candidate, batch, vector)

    assert torch.allclose(flatten_tree(output), expected)
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert reference_result.measurements["damping_min"] == pytest.approx(0.25)


def test_block_inverse_metric_schedule_must_match_representation() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([0.25], dtype=torch.float64),
        "b": torch.tensor([-0.75, 0.5], dtype=torch.float64),
    }
    operator = ops.inverse_metric(
        "inverse",
        "blocks",
        aggregation="sum",
        representation={"kind": "block_diagonal", "block_schedule": "module_blocks"},
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = vpx.Candidate(
        "inverse",
        "module-blocks",
        {
            **inverse_metric_settings("blockwise_solve"),
            "inverse_metric.block_schedule": "module_blocks",
        },
        admission_status="passed",
    )
    output = factory(candidate, {"metric_blocks": blocks}, vector)()
    expected = torch.linalg.solve(
        torch.block_diag(*blocks) + 0.25 * torch.eye(3, dtype=torch.float64),
        flatten_tree(vector),
    )

    assert torch.allclose(flatten_tree(output), expected)

    with pytest.raises(
        vp.MaterializationError,
        match=r"inverse_metric[.]block_schedule.*representation[.]block_schedule",
    ):
        factory(
            vpx.Candidate(
                "inverse",
                "custom-blocks",
                {
                    **inverse_metric_settings("blockwise_solve"),
                    "inverse_metric.block_schedule": "custom_blocks",
                },
                admission_status="passed",
            ),
            {"metric_blocks": blocks},
            vector,
        )


def test_low_rank_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = {
        "basis": torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        "diagonal": torch.tensor([4.0, 5.0], dtype=torch.float64),
    }
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    basis = factors["basis"]
    diagonal = factors["diagonal"]
    dense_matrix = basis @ basis.T + torch.diag(diagonal)
    expected = dense_matrix @ vector["w"]
    batch = {"low_rank_factors": factors}

    for path in ("factorized_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_low_rank_streaming_metric_uses_streaming_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    def forbidden_low_rank_metric_multiply(
        batch: Mapping[str, object],
        vector_tree: object,
        settings: Mapping[str, object],
    ) -> object:
        assert batch
        assert vector_tree
        assert settings
        message = "streaming_multiply called factorized low-rank path"
        raise AssertionError(message)

    monkeypatch.setattr(
        runtime_module,
        "_low_rank_metric_multiply",
        forbidden_low_rank_metric_multiply,
    )
    output = factory(
        vpx.Candidate(
            "metric",
            "streaming",
            metric_settings("streaming_multiply"),
            admission_status="passed",
        ),
        {"low_rank_factors": factors},
        vector,
    )()
    basis = factors["basis"]
    diagonal = factors["diagonal"]
    expected = (basis @ basis.T + torch.diag(diagonal)) @ vector["w"]

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_low_rank_inverse_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = {
        "basis": torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        "diagonal": torch.tensor([4.0, 5.0], dtype=torch.float64),
    }
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 5.0,
        },
    )
    basis = factors["basis"]
    diagonal = factors["diagonal"]
    dense_matrix = basis @ basis.T + torch.diag(diagonal)
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )
    batch = {"low_rank_factors": factors}

    for path in ("factorized_solve", "woodbury_low_rank_solve"):
        candidate = vpx.Candidate(
            "inverse",
            path,
            inverse_metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)
        assert reference_result.measurements["damping_min"] == pytest.approx(0.25)


def test_kfac_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = KFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "kfac",
        aggregation="sum",
        representation=kfac_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    dense_matrix = torch.kron(factors["w_left"], factors["w_right"])
    expected = dense_matrix @ vector["w"].reshape(-1)
    batch = {"kfac_factors": factors}

    for path in ("factorized_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(flatten_tree(output), expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_kfac_streaming_metric_uses_streaming_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = KFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "kfac",
        aggregation="sum",
        representation=kfac_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    def forbidden_kfac_metric_multiply(
        operator: vpx.OperatorSpec,
        batch: Mapping[str, object],
        vector_tree: object,
        settings: Mapping[str, object],
    ) -> object:
        assert operator.family == "metric"
        assert batch
        assert vector_tree
        assert settings
        message = "streaming_multiply called factorized KFAC path"
        raise AssertionError(message)

    monkeypatch.setattr(
        runtime_module,
        "_kfac_metric_multiply",
        forbidden_kfac_metric_multiply,
    )
    output = factory(
        vpx.Candidate(
            "metric",
            "streaming",
            metric_settings("streaming_multiply"),
            admission_status="passed",
        ),
        {"kfac_factors": factors},
        vector,
    )()
    expected = torch.kron(factors["w_left"], factors["w_right"]) @ vector["w"].reshape(
        -1
    )

    torch.testing.assert_close(flatten_tree(output), expected)


def test_kfac_inverse_metric_factorized_solve_matches_dense_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = KFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "kfac",
        aggregation="sum",
        representation=kfac_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 20.0,
        },
    )
    dense_matrix = torch.kron(factors["w_left"], factors["w_right"])
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(4, dtype=torch.float64),
        vector["w"].reshape(-1),
    )
    batch = {"kfac_factors": factors}
    candidate = vpx.Candidate(
        "inverse",
        "factorized",
        inverse_metric_settings("factorized_solve"),
        admission_status="passed",
    )
    output = factory(candidate, batch, vector)()
    reference_result = check(candidate, batch, vector)

    assert torch.allclose(flatten_tree(output), expected)
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert reference_result.measurements["damping_min"] == pytest.approx(0.25)


def test_ekfac_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = EKFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "ekfac",
        aggregation="sum",
        representation=ekfac_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    expected = ekfac_dense_matrix(factors) @ vector["w"].reshape(-1)

    for path in ("factorized_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, factors, vector)()
        reference_result = check(candidate, factors, vector)

        torch.testing.assert_close(flatten_tree(output), expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_ekfac_inverse_metric_factorized_solve_matches_dense_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = EKFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    damping = 0.25
    operator = ops.inverse_metric(
        "inverse",
        "ekfac",
        aggregation="sum",
        representation=ekfac_metric_representation(),
        damping=damping,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 20.0,
        },
    )
    dense_matrix = ekfac_dense_matrix(factors)
    expected = torch.linalg.solve(
        dense_matrix + damping * torch.eye(4, dtype=torch.float64),
        vector["w"].reshape(-1),
    )
    candidate = vpx.Candidate(
        "inverse",
        "factorized",
        inverse_metric_settings("factorized_solve"),
        admission_status="passed",
    )
    output = factory(candidate, factors, vector)()
    reference_result = check(candidate, factors, vector)

    torch.testing.assert_close(flatten_tree(output), expected)
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)


def test_ekfac_inverse_metric_per_group_damping_matches_dense_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = EKFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    damping = {"w": 0.25}
    operator = dataclasses.replace(
        ops.inverse_metric(
            "inverse",
            "ekfac",
            aggregation="sum",
            representation=ekfac_metric_representation(),
            damping=0.0,
        ),
        semantics={
            "damping": damping,
            "damping_kind": "per_group",
            "damping_value": damping,
            "representation": ekfac_metric_representation(),
        },
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 20.0,
        },
    )
    dense_matrix = ekfac_dense_matrix(factors)
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(4, dtype=torch.float64),
        vector["w"].reshape(-1),
    )
    candidate = vpx.Candidate(
        "inverse",
        "factorized",
        inverse_metric_settings("factorized_solve"),
        admission_status="passed",
    )
    output = factory(candidate, factors, vector)()
    reference_result = check(candidate, factors, vector)

    torch.testing.assert_close(flatten_tree(output), expected)
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)
    assert reference_result.measurements["damping_min"] == pytest.approx(0.25)


def test_ekfac_closed_form_square_root_paths_match_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = EKFACMetricData.factors()
    vector = {"w": torch.tensor([[0.25, -0.75], [0.5, 1.25]], dtype=torch.float64)}
    damping = 0.25
    settings = {"sqrt_metric.factor_path": "closed_form_factor_square_root"}
    sqrt_result, inverse_sqrt_result = metric_square_root_pair_results(
        ops.sqrt_metric(
            "sqrt_metric",
            "ekfac",
            aggregation="sum",
            representation=ekfac_metric_representation(),
        ),
        ops.inverse_sqrt_metric(
            "inverse_sqrt_metric",
            "ekfac",
            aggregation="sum",
            representation=ekfac_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch=factors,
        settings=settings,
        sqrt_vector=vector,
        inverse_vector=vector,
    )

    torch.testing.assert_close(
        tensor_mapping(sqrt_result)["w"],
        ekfac_square_root_product(factors, vector["w"], inverse=False),
    )
    torch.testing.assert_close(
        tensor_mapping(inverse_sqrt_result)["w"],
        ekfac_square_root_product(
            factors,
            vector["w"],
            inverse=True,
            damping=damping,
        ),
    )


def test_ekfac_metric_inner_factored_gram_matches_reference() -> None:
    params = {"w": torch.zeros((2, 2), dtype=torch.float64)}
    factors = EKFACMetricData.factors()
    left = {"w": torch.tensor([[1.5, -0.5], [0.75, 0.25]], dtype=torch.float64)}
    right = {"w": torch.tensor([[0.25, 2.0], [-1.0, 0.5]], dtype=torch.float64)}
    damping = 0.25
    metric_result = metric_inner_factored_gram_result(
        ops.metric_inner(
            "metric_inner",
            "ekfac",
            aggregation="sum",
            representation=ekfac_metric_representation(),
        ),
        params=params,
        batch=factors,
        left=left,
        right=right,
    )
    inverse_result = metric_inner_factored_gram_result(
        ops.inverse_metric_inner(
            "inverse_metric_inner",
            "ekfac",
            aggregation="sum",
            representation=ekfac_metric_representation(),
            damping=damping,
        ),
        params=params,
        batch=factors,
        left=left,
        right=right,
    )
    dense_matrix = ekfac_dense_matrix(factors)
    damped = dense_matrix + damping * torch.eye(4, dtype=torch.float64)

    torch.testing.assert_close(
        metric_result,
        flatten_tree(left) @ (dense_matrix @ flatten_tree(right)),
    )
    torch.testing.assert_close(
        inverse_result,
        flatten_tree(left) @ torch.linalg.solve(damped, flatten_tree(right)),
    )


def test_ggn_metric_paths_match_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = GGNMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "ggn",
        aggregation="sum",
        representation=ggn_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
        },
    )
    jacobian = factors["jacobian"]
    loss_hessian = factors["loss_hessian"]
    dense_matrix = jacobian.T @ loss_hessian @ jacobian
    expected = dense_matrix @ vector["w"]
    batch = {"ggn_factors": factors}

    for path in ("factorized_multiply", "streaming_multiply"):
        candidate = vpx.Candidate(
            "metric",
            path,
            metric_settings(path),
            admission_status="passed",
        )
        output = factory(candidate, batch, vector)()
        reference_result = check(candidate, batch, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["psd_violation"] == pytest.approx(0.0)


def test_ggn_streaming_metric_uses_streaming_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = GGNMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.metric(
        "metric",
        "ggn",
        aggregation="sum",
        representation=ggn_metric_representation(),
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    def forbidden_ggn_metric_multiply(
        batch: Mapping[str, object],
        vector_tree: object,
        settings: Mapping[str, object],
    ) -> object:
        assert batch
        assert vector_tree
        assert settings
        message = "streaming_multiply called factorized GGN path"
        raise AssertionError(message)

    monkeypatch.setattr(
        runtime_module,
        "_ggn_metric_multiply",
        forbidden_ggn_metric_multiply,
    )
    output = factory(
        vpx.Candidate(
            "metric",
            "streaming",
            metric_settings("streaming_multiply"),
            admission_status="passed",
        ),
        {"ggn_factors": factors},
        vector,
    )()
    jacobian = factors["jacobian"]
    loss_hessian = factors["loss_hessian"]
    expected = jacobian.T @ loss_hessian @ jacobian @ vector["w"]

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_ggn_inverse_metric_factorized_solve_matches_dense_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = GGNMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "ggn",
        aggregation="sum",
        representation=ggn_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    batch = {"ggn_factors": factors}
    candidate = vpx.Candidate(
        "inverse",
        "factorized",
        inverse_metric_settings("factorized_solve"),
        admission_status="passed",
    )

    output = factory(candidate, batch, vector)()

    jacobian = factors["jacobian"]
    loss_hessian = factors["loss_hessian"]
    dense_matrix = jacobian.T @ loss_hessian @ jacobian
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )

    torch.testing.assert_close(tree_leaves(output)[0], expected)


def test_inverse_metric_cg_dense_preconditioners_match_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "damping_min": 0.1,
            "condition_number_max": 5.0,
        },
    )
    expected = torch.linalg.solve(
        matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )

    for preconditioner in ("none", "diagonal"):
        candidate = vpx.Candidate(
            "inverse",
            preconditioner,
            {
                "inverse_metric.solve_path": "conjugate_gradient",
                "inverse_metric.iteration_budget": 2,
                "inverse_metric.preconditioner": preconditioner,
                "metric.multiply_path": "dense_matmul",
            },
            admission_status="passed",
        )
        output = factory(candidate, {"metric_matrix": matrix}, vector)()
        reference_result = check(candidate, {"metric_matrix": matrix}, vector)

        assert torch.allclose(tree_leaves(output)[0], expected)
        assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)
        assert reference_result.measurements["inverse_residual"] == pytest.approx(0.0)


def test_inverse_metric_matrix_free_preconditioner_uses_sibling_product() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    calls = []
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.25,
    )

    def preconditioner(
        batch: vpx.Batch,
        residual: vpx.TensorTree,
    ) -> vpx.TensorTree:
        assert batch["metric_matrix"] is matrix
        calls.append(tree_leaves(residual)[0].clone())

        return {"w": tensor_mapping(residual)["w"].clone()}

    runtime = runtime_module.standard_runtime_with_matrix_free_bindings(
        vpx.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=(),
            thresholds={
                "max_abs_diff": 1e-12,
                "max_rel_diff": 1e-12,
                "symmetry_max_abs_diff": 1e-12,
                "psd_violation": 1e-12,
                "inverse_residual": 1e-12,
            },
            objective_signature={"inverse": "matrix-free-preconditioner"},
            axis_registry=None,
        ),
        bindings={"identity_preconditioner": preconditioner},
        binding_signature={"identity_preconditioner": {"kind": "identity"}},
    )
    factory = runtime.operation_factory

    with pytest.raises(
        vp.MaterializationError,
        match=r"requires inverse_metric\.preconditioner_product",
    ):
        factory(
            vpx.Candidate(
                "inverse",
                "matrix-free-preconditioner-missing-product",
                {
                    "inverse_metric.solve_path": "conjugate_gradient",
                    "inverse_metric.iteration_budget": 2,
                    "inverse_metric.preconditioner": "matrix_free",
                    "metric.multiply_path": "dense_matmul",
                },
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            vector,
        )()

    result = factory(
        passed_candidate(
            "inverse",
            "matrix-free-preconditioner",
            {
                "inverse_metric.solve_path": "conjugate_gradient",
                "inverse_metric.iteration_budget": 2,
                "inverse_metric.preconditioner": "matrix_free",
                "inverse_metric.preconditioner_product": "identity_preconditioner",
                "metric.multiply_path": "dense_matmul",
            },
        ),
        {"metric_matrix": matrix},
        vector,
    )()
    expected = torch.linalg.solve(
        matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )

    assert calls
    torch.testing.assert_close(tree_leaves(result)[0], expected)

    unbound_factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(
        vp.MaterializationError,
        match="requires selected sibling product",
    ):
        unbound_factory(
            vpx.Candidate(
                "inverse",
                "matrix-free-preconditioner",
                {
                    "inverse_metric.solve_path": "conjugate_gradient",
                    "inverse_metric.iteration_budget": 2,
                    "inverse_metric.preconditioner": "matrix_free",
                    "inverse_metric.preconditioner_product": "identity_preconditioner",
                    "metric.multiply_path": "dense_matmul",
                },
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            vector,
        )()


def test_inverse_metric_cg_factor_reuse_does_not_filter_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_count_nonzero(value: torch.Tensor) -> torch.Tensor:
        _ = value
        message = "batched CG must keep a fixed RHS shape"

        raise AssertionError(message)

    monkeypatch.setattr(torch, "count_nonzero", forbidden_count_nonzero)
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.eye(2, dtype=torch.float64)
    vector = {
        "w": torch.tensor(
            [[0.0, 0.0], [1.0, 2.0]],
            dtype=torch.float64,
        )
    }
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    result = factory(
        vpx.Candidate(
            "inverse",
            "cg-factor-reuse",
            {
                **inverse_metric_settings("conjugate_gradient"),
                "inverse_metric.iteration_budget": 2,
                "inverse_metric.preconditioner": "none",
                "inverse_metric.factor_reuse": "reuse_factor_across_rhs",
                "inverse_metric.multi_rhs": "block",
                "metric.multiply_path": "dense_matmul",
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"])


def test_inverse_metric_cg_requires_metric_accumulation_for_non_dense_inner_path() -> (
    None
):
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = vpx.Candidate(
        "inverse",
        "missing-accumulation",
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 1,
            "inverse_metric.preconditioner": "none",
            "metric.multiply_path": "factorized_multiply",
        },
        admission_status="passed",
    )

    with pytest.raises(
        vp.MaterializationError, match=r"metric\.accumulation is required"
    ):
        factory(candidate, {"low_rank_factors": factors}, vector)


def test_inverse_metric_cg_uses_declared_inner_metric_path() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    candidate = vpx.Candidate(
        "inverse",
        "blockwise-inner",
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 1,
            "inverse_metric.preconditioner": "none",
            **metric_settings("blockwise_multiply"),
        },
        admission_status="passed",
    )
    operation = factory(candidate, {"metric_matrix": matrix}, vector)

    with pytest.raises(vp.MaterializationError, match="not supported by path"):
        operation()


def test_inverse_metric_cg_stops_at_declared_tol() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    vector = {"w": torch.ones(2, dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
        tol=0.7,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    output = factory(
        vpx.Candidate(
            "inverse",
            "cg-tol",
            {
                **inverse_metric_settings("conjugate_gradient"),
                "inverse_metric.iteration_budget": 4,
                "inverse_metric.preconditioner": "none",
                **metric_settings("dense_matmul"),
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )()
    one_step = torch.tensor([0.4, 0.4], dtype=torch.float64)
    exact = torch.linalg.solve(matrix, vector["w"])
    result = tree_leaves(output)[0]

    torch.testing.assert_close(result, one_step)
    assert not torch.allclose(result, exact)


def test_inverse_metric_tol_requires_conjugate_gradient_path() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.eye(2, dtype=torch.float64)
    vector = {"w": torch.ones(2, dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
        tol=1e-4,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(
        vp.MaterializationError, match="tol requires conjugate_gradient"
    ):
        factory(
            vpx.Candidate(
                "inverse",
                "direct-solve",
                inverse_metric_settings("dense_solve"),
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            vector,
        )


def test_inverse_metric_reference_check_uses_declared_tol_threshold() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    vector = {"w": torch.ones(2, dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
            tol=0.7,
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 10.0,
            "max_rel_diff": 10.0,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "condition_number_max": 10.0,
        },
    )
    result = check(
        vpx.Candidate(
            "inverse",
            "cg-tol",
            {
                **inverse_metric_settings("conjugate_gradient"),
                "inverse_metric.iteration_budget": 4,
                "inverse_metric.preconditioner": "none",
                **metric_settings("dense_matmul"),
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )

    assert result.thresholds["inverse_residual"] == pytest.approx(0.7)
    assert result.measurements["inverse_residual"] == pytest.approx(0.6)


def test_inverse_metric_inner_cg_uses_declared_tol_before_reduction() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    left = {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}
    right = {"w": torch.ones(2, dtype=torch.float64)}
    operator = ops.inverse_metric_inner(
        "inner",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
        tol=0.7,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    output = factory(
        vpx.Candidate(
            "inner",
            "cg-tol",
            {
                "inverse_metric_inner.reduction_path": "solve_then_reduce",
                "inverse_metric_inner.multi_rhs": "single_column",
                **inverse_metric_settings("conjugate_gradient"),
                "inverse_metric.iteration_budget": 4,
                "inverse_metric.preconditioner": "none",
                **metric_settings("dense_matmul"),
            },
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        (left, right),
    )()

    torch.testing.assert_close(output, torch.tensor(0.4, dtype=torch.float64))


def test_inverse_metric_inner_tol_requires_solve_then_reduce() -> None:
    params = {"w": torch.zeros(2, dtype=torch.float64)}
    matrix = torch.eye(2, dtype=torch.float64)
    vector = (
        {"w": torch.ones(2, dtype=torch.float64)},
        {"w": torch.ones(2, dtype=torch.float64)},
    )
    operator = ops.inverse_metric_inner(
        "inner",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.0,
        tol=1e-4,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="tol requires solve_then_reduce"):
        factory(
            vpx.Candidate(
                "inner",
                "sqrt-reduce",
                {
                    "inverse_metric_inner.reduction_path": "sqrt_apply_reduce",
                    "inverse_metric_inner.multi_rhs": "single_column",
                    "sqrt_metric.factor_path": "cholesky_factor",
                },
                admission_status="passed",
            ),
            {"metric_matrix": matrix},
            vector,
        )


def test_inverse_metric_cg_block_preconditioner_matches_reference() -> None:
    params = {
        "a": torch.tensor([1.0], dtype=torch.float64),
        "b": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    blocks = (
        torch.tensor([[4.0]], dtype=torch.float64),
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    vector = {
        "a": torch.tensor([0.25], dtype=torch.float64),
        "b": torch.tensor([-0.75, 0.5], dtype=torch.float64),
    }
    operator = ops.inverse_metric(
        "inverse",
        "blocks",
        aggregation="sum",
        representation=block_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    dense_matrix = torch.block_diag(*blocks)
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(3, dtype=torch.float64),
        flatten_tree(vector),
    )
    candidate = vpx.Candidate(
        "inverse",
        "cg-block",
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 3,
            "inverse_metric.preconditioner": "block_diagonal",
            **metric_settings("blockwise_multiply"),
        },
        admission_status="passed",
    )
    output = factory(candidate, {"metric_blocks": blocks}, vector)()

    assert torch.allclose(flatten_tree(output), expected)


def test_inverse_metric_cg_factorized_preconditioner_matches_reference() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factors = LowRankMetricData.factors()
    vector = {"w": torch.tensor([0.25, -0.75], dtype=torch.float64)}
    operator = ops.inverse_metric(
        "inverse",
        "low_rank",
        aggregation="sum",
        representation=low_rank_metric_representation(),
        damping=0.25,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    basis = factors["basis"]
    diagonal = factors["diagonal"]
    dense_matrix = basis @ basis.T + torch.diag(diagonal)
    expected = torch.linalg.solve(
        dense_matrix + 0.25 * torch.eye(2, dtype=torch.float64),
        vector["w"],
    )
    candidate = vpx.Candidate(
        "inverse",
        "cg-factorized",
        {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": 1,
            "inverse_metric.preconditioner": "factorized_metric",
            **metric_settings("factorized_multiply"),
        },
        admission_status="passed",
    )
    output = factory(candidate, {"low_rank_factors": factors}, vector)()

    assert torch.allclose(tree_leaves(output)[0], expected)


def test_inverse_metric_reference_check_rejects_indefinite_metric() -> None:
    params = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    check = vpx.standard_reference_check(
        ops.inverse_metric(
            "inverse",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
            damping=0.0,
        ),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inverse_residual": 1e-12,
            "condition_number_max": 10.0,
        },
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "inverse",
                "row",
                inverse_metric_settings(),
                admission_status="passed",
            ),
            {
                "metric_matrix": torch.diag(
                    torch.tensor([1.0, -0.1], dtype=torch.float64)
                )
            },
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_rejects_nonsymmetric_loss_hessian() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 2.0 * params["w"][0]))

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
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
            vpx.Candidate(
                "ggn",
                "row",
                ggn_dense_kernel_settings(),
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.tensor(
                    [[1.0, 2.0], [0.0, 1.0]],
                    dtype=torch.float64,
                ),
                "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_rejects_indefinite_loss_hessian() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 2.0 * params["w"][0]))

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "ggn",
                "row",
                ggn_dense_kernel_settings(),
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.diag(
                    torch.tensor(
                        [1.0, -0.1],
                        dtype=torch.float64,
                    )
                ),
                "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_uses_jvp_hessian_vjp_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        multiplier = (
            2.0 if context.settings.get("ggn.jvp_path") == "torch_func_jvp" else 1.0
        )

        return multiplier * torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "ggn",
                "row",
                {
                    **ggn_settings("torch_func_linearize"),
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.eye(2, dtype=torch.float64),
                "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_cross_checks_jvp_path_with_dense_anchor() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        multiplier = (
            2.0 if context.settings.get("ggn.jvp_path") == "torch_func_jvp" else 1.0
        )

        return multiplier * torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
        function_objectives={"model_output": function},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "ggn",
                "row",
                {
                    **ggn_settings("forward_ad_dual"),
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.eye(2, dtype=torch.float64),
                "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_ggnvp_reference_check_records_dense_anchor_errors() -> None:
    params = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is not None
        assert context.family == "ggn"

        return torch.stack((params["w"][0], 3.0 * params["w"][0]))

    check = vpx.standard_reference_check(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "psd_violation": 1e-12,
            "inner_abs_diff": 1e-12,
        },
        function_objectives={"model_output": function},
    )
    result = check(
        vpx.Candidate(
            "ggn",
            "jvp",
            {
                **ggn_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
            },
            admission_status="passed",
        ),
        {
            "loss_hessian": torch.eye(2, dtype=torch.float64),
            "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )
    linearize_result = check(
        vpx.Candidate(
            "ggn",
            "linearize",
            {
                **ggn_settings("torch_func_linearize"),
                **torch_func_settings(requires_forward_ad=True),
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
            },
            admission_status="passed",
        ),
        {
            "loss_hessian": torch.eye(2, dtype=torch.float64),
            "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )
    forward_ad_result = check(
        vpx.Candidate(
            "ggn",
            "forward-ad",
            {
                **ggn_settings("forward_ad_dual"),
                **torch_func_settings(requires_forward_ad=True),
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
            },
            admission_status="passed",
        ),
        {
            "loss_hessian": torch.eye(2, dtype=torch.float64),
            "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
        },
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert result.measurements["dense_anchor_errors"] == {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }
    assert linearize_result.measurements["dense_anchor_errors"] == {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }
    assert forward_ad_result.measurements["dense_anchor_errors"] == {
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check(
            vpx.Candidate(
                "ggn",
                "zero-vector",
                {
                    **ggn_settings("torch_func_jvp"),
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.eye(2, dtype=torch.float64),
                "symmetry_vector": {"w": torch.tensor([2.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([0.0], dtype=torch.float64)},
        )

    with pytest.raises(vp.ReferenceFailedError, match="min_probe_norm"):
        check(
            vpx.Candidate(
                "ggn",
                "zero-symmetry-vector",
                {
                    **ggn_settings("torch_func_jvp"),
                    **torch_func_settings(requires_forward_ad=True),
                    "ggn.loss_hessian_path": "autodiff_loss_hvp",
                    "ggn.loss_hessian_kernel": "dense_global",
                },
                admission_status="passed",
            ),
            {
                "loss_hessian": torch.eye(2, dtype=torch.float64),
                "symmetry_vector": {"w": torch.tensor([0.0], dtype=torch.float64)},
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )
    assert result.measurements["inner_abs_diff"] == pytest.approx(0.0)
    assert linearize_result.measurements["inner_abs_diff"] == pytest.approx(0.0)
    assert forward_ad_result.measurements["inner_abs_diff"] == pytest.approx(0.0)


def test_fisher_references_use_declared_per_example_objectives() -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}

    def per_example_scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["normalization"] == pytest.approx(3.0)
        assert context.family in {"fisher", "empirical"}

        return torch.stack((
            params["w"][0],
            2.0 * params["w"][0] - params["w"][1],
            0.5 * params["w"][0] + 3.0 * params["w"][1],
        ))

    score_gradients = _float_rows(SCORE_GRADIENT_ROWS)
    wrong_gradients = torch.zeros_like(score_gradients)
    thresholds = {"max_abs_diff": 1e-12, "max_rel_diff": 1e-12}
    fisher_check = vpx.standard_reference_check(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        thresholds=thresholds,
        function_objectives={"scores": per_example_scores},
    )
    empirical_check = vpx.standard_reference_check(
        ops.empirical_fisher_vp(
            "empirical",
            "scores",
            aggregation="mean_per_example",
            example_loss_reduction="per_example",
            denominator="batch_normalization",
        ),
        params=params,
        buffers={},
        thresholds=thresholds,
        function_objectives={"scores": per_example_scores},
    )
    fisher_result = fisher_check(
        vpx.Candidate(
            "fisher",
            "row",
            fisher_settings("materialize_score_gradients"),
            admission_status="passed",
        ),
        {"score_gradients": score_gradients, "normalization": 3.0},
        vector,
    )
    empirical_result = empirical_check(
        vpx.Candidate(
            "empirical",
            "row",
            empirical_dense_settings(),
            admission_status="passed",
        ),
        {"per_example_gradients": score_gradients, "normalization": 3.0},
        vector,
    )

    assert fisher_result.measurements["max_abs_diff"] == pytest.approx(0.0)
    assert empirical_result.measurements["max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(vp.ReferenceFailedError):
        fisher_check(
            vpx.Candidate(
                "fisher",
                "row",
                fisher_settings("materialize_score_gradients"),
                admission_status="passed",
            ),
            {"score_gradients": wrong_gradients, "normalization": 3.0},
            vector,
        )

    with pytest.raises(vp.ReferenceFailedError):
        empirical_check(
            vpx.Candidate(
                "empirical",
                "row",
                empirical_dense_settings(),
                admission_status="passed",
            ),
            {"per_example_gradients": wrong_gradients, "normalization": 3.0},
            vector,
        )


@pytest.mark.parametrize(
    ("operator", "family", "settings", "batch_key"),
    [
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            fisher_settings("materialize_score_gradients"),
            "score_gradients",
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            sampled_fisher_settings("materialize_score_gradients"),
            "sampled_score_gradients",
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="batch_normalization",
            ),
            "empirical",
            empirical_dense_settings(),
            "per_example_gradients",
        ),
    ],
)
def test_numeric_loss_scaling_unscales_degree_two_fisher_family_output(
    operator: vpx.OperatorSpec,
    family: str,
    settings: Mapping[str, object],
    batch_key: str,
) -> None:
    params = {"w": torch.tensor([0.0, 0.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}
    score_gradients = _float_rows(SCORE_GRADIENT_ROWS)
    batch = {
        batch_key: score_gradients,
        "normalization": 3.0,
        "num_examples": 3.0,
    }
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        function_objectives={},
    )
    unscaled = factory(
        vpx.Candidate(
            family,
            "unscaled",
            settings,
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    scaled = factory(
        vpx.Candidate(
            family,
            "scaled",
            {
                **settings,
                **loss_scaling_settings(degree=2, scale=4.0),
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    assert torch.allclose(tree_leaves(scaled)[0], tree_leaves(unscaled)[0])


def test_empirical_fisher_vmap_path_matches_loop_path() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}
    batch = {
        "x": torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
        "y": torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64),
        "normalization": 3.0,
    }

    def per_example_losses(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "empirical"
        x_value = batch["x"]
        y_value = batch["y"]
        assert isinstance(x_value, torch.Tensor)
        assert isinstance(y_value, torch.Tensor)

        return (params["w"][0] * x_value - y_value).square()

    operator = empirical_fisher_mean("empirical", "losses")
    factory = empirical_loss_factory(params, per_example_losses)
    results = candidate_leaf_results(
        factory,
        "empirical",
        (
            (
                "loop",
                {
                    **empirical_grad_settings("torch_autograd_grad_loop"),
                    "schedule.per_example": "loop",
                },
            ),
            (
                "torch-func",
                {
                    **empirical_grad_settings("torch_func_grad"),
                    "schedule.per_example": "loop",
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
            (
                "materialized",
                {
                    **empirical_grad_settings("backward_materialized_grad"),
                    "schedule.per_example": "loop",
                },
            ),
            (
                "vmap",
                {
                    **empirical_grad_settings("vmap_grad"),
                    **empirical_per_example_vmap_settings(),
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
        ),
        batch,
        vector,
    )
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
        function_objectives={"losses": per_example_losses},
    )
    reference_result = check(
        passed_candidate(
            "empirical",
            "vmap-reference",
            {
                **empirical_grad_settings("vmap_grad"),
                **empirical_per_example_vmap_settings(),
                **torch_func_settings(requires_forward_ad=False),
            },
        ),
        batch,
        vector,
    )

    assert torch.allclose(results["vmap"], results["loop"])
    assert torch.allclose(results["torch-func"], results["loop"])
    assert torch.allclose(results["materialized"], results["loop"])
    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)

    for candidate_id, settings, message in (
        (
            "bad-loop",
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                "vectorization.vmap_chunk_size": 1,
            },
            r"vectorization\.vmap_chunk_size",
        ),
        (
            "missing-schedule",
            {
                **empirical_grad_settings("vmap_grad"),
                "batch.empirical_example_batch_size": 1,
                **torch_func_settings(requires_forward_ad=False),
            },
            r"schedule\.per_example",
        ),
        (
            "wrong-batch-size-key",
            {
                **empirical_grad_settings("vmap_grad"),
                "schedule.per_example": "vmap",
                "batch.fisher_sample_batch_size": 1,
                **torch_func_settings(requires_forward_ad=False),
            },
            r"batch\.fisher_sample_batch_size",
        ),
    ):
        with pytest.raises(vp.MaterializationError, match=message):
            run_passed_candidate(
                factory,
                "empirical",
                candidate_id,
                settings,
                batch,
                vector,
            )

    vector_axis_vmap_result = run_passed_candidate(
        factory,
        "empirical",
        "vector-axis-vmap",
        {
            **empirical_grad_settings("vmap_grad"),
            **empirical_per_example_vmap_settings(),
            **vmap_settings({"w": 0}, chunk_size=1),
            **torch_func_settings(requires_forward_ad=False),
        },
        batch,
        {"w": vector["w"].reshape(1, 1)},
    )

    torch.testing.assert_close(
        tree_leaves(vector_axis_vmap_result)[0],
        results["loop"].reshape(1, 1),
    )


def test_empirical_fisher_vmap_path_rejects_invalid_batch_shape() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}

    def per_example_losses(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert params
        assert buffers == {}
        assert context.family == "empirical"
        x_value = batch["x"]
        assert isinstance(x_value, torch.Tensor)

        return x_value * 0.0

    factory = empirical_loss_factory(params, per_example_losses)

    with pytest.raises(vp.MaterializationError, match="leading dimensions differ"):
        run_passed_candidate(
            factory,
            "empirical",
            "vmap",
            {
                **empirical_grad_settings("vmap_grad"),
                **empirical_per_example_vmap_settings(),
                **torch_func_settings(requires_forward_ad=False),
            },
            {
                "x": torch.tensor([1.0, 2.0], dtype=torch.float64),
                "y": torch.tensor([1.0], dtype=torch.float64),
                "normalization": 2.0,
            },
            vector,
        )


def test_per_example_schedule_rejects_non_batch_data_axis() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}

    def per_example_losses(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert params
        assert buffers == {}
        assert batch
        assert context.family == "empirical"

        return torch.tensor([1.0, 2.0], dtype=torch.float64)

    operator = dataclasses.replace(
        empirical_fisher_mean("empirical", "losses"),
        data_axis="sequence",
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        function_objectives={"losses": per_example_losses},
    )

    with pytest.raises(vp.MaterializationError, match="data_axis=batch"):
        run_passed_candidate(
            factory,
            "empirical",
            "vmap",
            {
                **empirical_grad_settings("vmap_grad"),
                **empirical_per_example_vmap_settings(),
                **torch_func_settings(requires_forward_ad=False),
            },
            {
                "x": torch.tensor([1.0, 2.0], dtype=torch.float64),
                "normalization": 2.0,
            },
            vector,
        )


def test_microbatch_accumulation_rejects_non_batch_data_axis() -> None:
    params = {"w": torch.tensor([0.4], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.7], dtype=torch.float64)}
    operator = dataclasses.replace(
        ops.gradient("gradient", "loss", aggregation="sum"),
        data_axis="sequence",
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="data_axis=batch"):
        factory(
            vpx.Candidate(
                "gradient",
                "microbatch",
                {
                    **gradient_settings(),
                    "schedule.gradient_accumulation": "microbatch_accumulate",
                    "batch.data_microbatch_size": 1,
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            vector,
        )()


def test_standard_operation_factory_runs_dense_metric_and_fisher_families() -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    buffers = {}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    score_gradients = _float_rows(SCORE_GRADIENT_ROWS)
    score_gradient_blocks = COLUMN_BLOCKS(score_gradients)
    sampled_score_gradients = _float_rows((*SCORE_GRADIENT_ROWS, (-1.0, 2.0)))
    sampled_score_gradient_blocks = COLUMN_BLOCKS(sampled_score_gradients)
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["w"][0] + 2.0 * params["w"][1],
            -params["w"][0] + params["w"][1],
        ))

    factories = {
        "ggn": standard_factory(
            params,
            buffers,
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            function_objectives={"model_output": function},
        ),
        "fisher": standard_factory(
            params,
            buffers,
            score_terms_fisher("fisher", "scores"),
        ),
        "sampled": standard_factory(
            params,
            buffers,
            score_terms_sampled_fisher("sampled", "sampled_scores"),
        ),
        "empirical": standard_factory(
            params,
            buffers,
            empirical_fisher_mean(
                "empirical",
                "scores",
                denominator="batch_normalization",
            ),
        ),
        "metric": standard_factory(
            params,
            buffers,
            ops.metric(
                "metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
            ),
        ),
        "inverse": standard_factory(
            params,
            buffers,
            ops.inverse_metric(
                "inverse",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
                damping=0.0,
            ),
        ),
    }
    flat_vector = vector["w"]
    jacobian = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    expected_ggn = jacobian.T @ (loss_hessian @ (jacobian @ flat_vector))
    expected_fisher = score_gradients.T @ (score_gradients @ flat_vector) / 3.0
    expected_sampled = (
        sampled_score_gradients.T @ (sampled_score_gradients @ flat_vector) / 4.0
    )

    cases = (
        (
            "ggn",
            "row",
            ggn_dense_kernel_settings(),
            {"loss_hessian": loss_hessian},
            expected_ggn,
        ),
        (
            "fisher",
            "row",
            fisher_settings("materialize_score_gradients"),
            {"score_gradients": score_gradients, "normalization": 3.0},
            expected_fisher,
        ),
        (
            "fisher",
            "blockwise",
            fisher_settings("blockwise_score_matrix"),
            {"score_gradient_blocks": score_gradient_blocks, "normalization": 3.0},
            expected_fisher,
        ),
        (
            "sampled",
            "row",
            sampled_fisher_settings("materialize_score_gradients"),
            {"sampled_score_gradients": sampled_score_gradients, "num_examples": 2},
            expected_sampled,
        ),
        (
            "sampled",
            "blockwise",
            sampled_fisher_settings("blockwise_score_matrix"),
            {
                "sampled_score_gradient_blocks": sampled_score_gradient_blocks,
                "num_examples": 2,
            },
            expected_sampled,
        ),
        (
            "empirical",
            "row",
            empirical_dense_settings(),
            {"per_example_gradients": score_gradients, "normalization": 3.0},
            expected_fisher,
        ),
        (
            "empirical",
            "blockwise",
            {"empirical_fisher.accumulation": "blockwise_gradient_matrix"},
            {
                "per_example_gradient_blocks": score_gradient_blocks,
                "normalization": 3.0,
            },
            expected_fisher,
        ),
        (
            "metric",
            "row",
            metric_settings(),
            {"metric_matrix": matrix},
            matrix @ flat_vector,
        ),
        (
            "inverse",
            "row",
            inverse_metric_settings(),
            {"metric_matrix": matrix},
            torch.linalg.solve(matrix, flat_vector),
        ),
    )

    for family, candidate_id, settings, product_batch, expected in cases:
        result = run_passed_candidate(
            factories[family],
            family,
            candidate_id,
            settings,
            product_batch,
            vector,
        )
        torch.testing.assert_close(tree_leaves(result)[0], expected)


def chunk_layer_surface() -> vpx.ParameterSurface:
    return vpx.ParameterSurface(
        names=("a", "b", "c"),
        shapes=((1,), (2,), (1,)),
        trainable=(True, True, True),
        layer_groups=(("a", "b"), ("c",)),
    )


@pytest.mark.parametrize(
    (
        "params",
        "parameter_surface",
        "vector",
        "score_gradients",
        "settings_key",
        "settings_value",
        "candidate_id",
        "expected_calls",
    ),
    [
        (
            {"w": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)},
            None,
            {"w": torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)},
            torch.tensor(
                [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]],
                dtype=torch.float64,
            ),
            "chunk.parameter_block_size",
            1,
            "parameter-blocks",
            [
                ((2, 1), (1,)),
                ((2, 1), (1,)),
                ((2, 1), (1,)),
                ((1, 2), (2,)),
                ((1, 2), (2,)),
                ((1, 2), (2,)),
            ],
        ),
        (
            {
                "a": torch.tensor([0.0], dtype=torch.float64),
                "b": torch.tensor([0.0, 0.0], dtype=torch.float64),
                "c": torch.tensor([0.0], dtype=torch.float64),
            },
            chunk_layer_surface(),
            {
                "a": torch.tensor([1.0], dtype=torch.float64),
                "b": torch.tensor([-2.0, 3.0], dtype=torch.float64),
                "c": torch.tensor([4.0], dtype=torch.float64),
            },
            torch.tensor(
                [[1.0, 2.0, -1.0, 0.5], [0.5, -3.0, 4.0, -2.0]],
                dtype=torch.float64,
            ),
            "chunk.layer_block_size",
            1,
            "layer-blocks",
            [
                ((2, 3), (3,)),
                ((2, 1), (1,)),
                ((3, 2), (2,)),
                ((1, 2), (2,)),
            ],
        ),
    ],
)
def test_chunk_size_chunks_score_matrix_product(
    monkeypatch: pytest.MonkeyPatch,
    params: vpx.ParameterTree,
    parameter_surface: vpx.ParameterSurface | None,
    vector: vpx.TensorTree,
    score_gradients: torch.Tensor,
    settings_key: str,
    settings_value: int,
    candidate_id: str,
    expected_calls: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> None:
    flat_vector = flatten_tree(vector)
    expected = score_gradients.T @ (score_gradients @ flat_vector) / 2.0
    calls = patch_matmul_call_recorder(monkeypatch, settings_key, settings_value)
    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        parameter_surface=parameter_surface,
    )
    result = factory(
        vpx.Candidate(
            "fisher",
            candidate_id,
            {
                **fisher_settings("materialize_score_gradients"),
                settings_key: settings_value,
            },
            admission_status="passed",
        ),
        {"score_gradients": score_gradients, "normalization": 2.0},
        vector,
    )()

    torch.testing.assert_close(flatten_tree(result), expected)
    assert calls == expected_calls


@pytest.mark.parametrize(
    (
        "params",
        "parameter_surface",
        "vector",
        "output_matrix",
        "settings_key",
        "settings_value",
        "candidate_id",
        "expected_calls",
    ),
    [
        (
            {"w": torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)},
            None,
            {"w": torch.tensor([1.0, 0.5, -2.0], dtype=torch.float64)},
            torch.tensor(
                [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]],
                dtype=torch.float64,
            ),
            "chunk.parameter_block_size",
            2,
            "parameter-blocks",
            [
                ((2, 2), (2,)),
                ((2, 1), (1,)),
                ((2, 2), (2,)),
                ((2, 2), (2,)),
                ((1, 2), (2,)),
            ],
        ),
        (
            {
                "a": torch.tensor([0.5], dtype=torch.float64),
                "b": torch.tensor([-1.0, 2.0], dtype=torch.float64),
                "c": torch.tensor([1.5], dtype=torch.float64),
            },
            chunk_layer_surface(),
            {
                "a": torch.tensor([1.0], dtype=torch.float64),
                "b": torch.tensor([0.5, -2.0], dtype=torch.float64),
                "c": torch.tensor([3.0], dtype=torch.float64),
            },
            torch.tensor(
                [[1.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 2.0]],
                dtype=torch.float64,
            ),
            "chunk.layer_block_size",
            1,
            "layer-blocks",
            [
                ((2, 3), (3,)),
                ((2, 1), (1,)),
                ((2, 2), (2,)),
                ((3, 2), (2,)),
                ((1, 2), (2,)),
            ],
        ),
    ],
)
def test_chunk_size_chunks_dense_ggn_transpose_product(
    monkeypatch: pytest.MonkeyPatch,
    params: vpx.ParameterTree,
    parameter_surface: vpx.ParameterSurface | None,
    vector: vpx.TensorTree,
    output_matrix: torch.Tensor,
    settings_key: str,
    settings_value: int,
    candidate_id: str,
    expected_calls: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> None:
    loss_hessian = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float64)
    flat_vector = flatten_tree(vector)
    expected = output_matrix.T @ (loss_hessian @ (output_matrix @ flat_vector))
    calls = patch_matmul_call_recorder(monkeypatch, settings_key, settings_value)

    def function(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return output_matrix @ flatten_tree(params)

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        parameter_surface=parameter_surface,
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            candidate_id,
            {
                "ggn.loss_hessian_path": "autodiff_loss_hvp",
                "ggn.loss_hessian_kernel": "dense_global",
                settings_key: settings_value,
            },
            admission_status="passed",
        ),
        {"loss_hessian": loss_hessian},
        vector,
    )()

    torch.testing.assert_close(flatten_tree(result), expected)
    assert calls == expected_calls


def test_parameter_block_size_rejects_non_dense_parameter_matrix_path() -> None:
    params = {"w": torch.tensor([0.0, 0.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="parameter-column"):
        factory(
            vpx.Candidate(
                "fisher",
                "bad-parameter-blocks",
                {
                    **fisher_settings("blockwise_score_matrix"),
                    "chunk.parameter_block_size": 1,
                },
                admission_status="passed",
            ),
            {"score_gradient_blocks": (torch.eye(2, dtype=torch.float64),)},
            vector,
        )


def test_layer_block_size_requires_declared_layer_groups() -> None:
    params = {"w": torch.tensor([0.0, 0.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="declared layer_groups"):
        factory(
            vpx.Candidate(
                "fisher",
                "missing-layer-groups",
                {
                    **fisher_settings("materialize_score_gradients"),
                    "chunk.layer_block_size": 1,
                },
                admission_status="passed",
            ),
            {
                "score_gradients": torch.eye(2, dtype=torch.float64),
                "normalization": 1.0,
            },
            vector,
        )


def test_parameter_and_layer_block_sizes_are_mutually_exclusive() -> None:
    params = {"w": torch.tensor([0.0, 0.0], dtype=torch.float64)}
    parameter_surface = vpx.ParameterSurface(
        names=("w",),
        shapes=((2,),),
        trainable=(True,),
        layer_groups=(("w",),),
    )
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        parameter_surface=parameter_surface,
    )

    with pytest.raises(vp.MaterializationError, match="cannot both be set"):
        factory(
            vpx.Candidate(
                "fisher",
                "conflicting-chunks",
                {
                    **fisher_settings("materialize_score_gradients"),
                    "chunk.parameter_block_size": 1,
                    "chunk.layer_block_size": 1,
                },
                admission_status="passed",
            ),
            {
                "score_gradients": torch.eye(2, dtype=torch.float64),
                "normalization": 1.0,
            },
            vector,
        )


def test_hvp_row_batch_size_executes_reverse_rows() -> None:
    params = {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.5, -0.5], dtype=torch.float64)}
    result = run_quadratic_hvp(
        "row-batched",
        {
            **hvp_settings("reverse_over_reverse"),
            "batch.hvp_row_batch_size": 1,
        },
        params,
        {"scale": 3.0},
        vector,
    )

    torch.testing.assert_close(
        tree_leaves(result)[0],
        6.0 * vector["w"],
    )


def test_hvp_row_batch_size_rejects_non_reverse_path() -> None:
    factory = quadratic_hvp_factory({"w": torch.tensor([1.0], dtype=torch.float64)})

    with pytest.raises(vp.MaterializationError, match="reverse_over_reverse"):
        factory(
            passed_candidate(
                "hvp",
                "bad-row-batch",
                {
                    **hvp_settings("jvp_grad"),
                    **torch_func_settings(requires_forward_ad=True),
                    "batch.hvp_row_batch_size": 1,
                },
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_fisher_blockwise_vmap_vectorization_runs_batched_vectors() -> None:
    params = {
        "a": torch.zeros(2, dtype=torch.float64),
        "b": torch.zeros((2, 1), dtype=torch.float64),
    }
    vector = {
        "b": torch.tensor(
            [[[3.0], [4.0]], [[7.0], [8.0]]],
            dtype=torch.float64,
        ),
        "a": torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float64),
    }
    score_gradients = torch.eye(4, dtype=torch.float64)
    score_gradient_blocks = (
        score_gradients[:, :2],
        score_gradients[:, 2:],
    )
    vector_settings = vmap_settings({"b": 0, "a": 0}, chunk_size=1)
    vector_admission = torch_func_settings(requires_forward_ad=False)
    flat_vectors = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )
    fisher_factory = vpx.standard_operation_factory(
        score_terms_fisher_sum("fisher", "scores"),
        params=params,
        buffers={},
    )
    result = run_passed_candidate(
        fisher_factory,
        "fisher",
        "blockwise-vmap",
        {
            **fisher_settings("blockwise_score_matrix"),
            **vector_settings,
            **vector_admission,
        },
        {"score_gradient_blocks": score_gradient_blocks, "normalization": 1.0},
        vector,
    )

    assert_two_leaf_batched_map(result, flat_vectors)


def test_fisher_vector_vmap_composes_with_per_example_vmap_score_path() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([[3.0], [5.0]], dtype=torch.float64)}
    batch = {
        "coeff": torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
        "normalization": 3.0,
    }

    def scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "fisher"

        return params["w"] * batch["coeff"]

    factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
        function_objectives={"scores": scores},
    )
    result = factory(
        vpx.Candidate(
            "fisher",
            "score-vmap-vector-vmap",
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="vmap_grad",
                ),
                **fisher_per_example_vmap_settings(batch_size=2),
                **vmap_settings({"w": 0}, chunk_size=1),
                **torch_func_settings(requires_forward_ad=False),
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    expected = vector["w"] * ((2.0**2 + 4.0**2 + 6.0**2) / 3.0)

    torch.testing.assert_close(tree_leaves(result)[0], expected)


def test_standard_metric_materializer_returns_metric_object(tmp_path: Path) -> None:
    model = TwoParameterModule()
    params = {"w": model.w.detach().clone()}
    operator = ops.metric(
        "metric",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
    )
    candidates = (
        vpx.Candidate(
            "metric",
            "dense",
            metric_settings(),
            changed_axes=("metric.multiply_path",),
        ),
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=DenseMetricData(),
        operator=operator,
        vectors=TwoParameterVectorProvider(),
        target=cpu_target(),
        runtime=vpx.standard_runtime_config(
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
            axis_registry=vpx.standard_axis_registry(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, name="metric")
    batch = {"metric_matrix": DenseMetricData.matrix}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    right = {"w": torch.tensor([-1.0, 0.5], dtype=torch.float64)}

    assert isinstance(selected, vpx.StandardMetricOperator)
    assert torch.allclose(
        tree_leaves(selected(batch, vector))[0],
        DenseMetricData.matrix @ vector["w"],
    )
    assert torch.allclose(
        tree_leaves(selected.multiply(batch, vector))[0],
        DenseMetricData.matrix @ vector["w"],
    )
    with pytest.raises(vp.MaterializationError, match="inverse_metric selection"):
        selected.inverse_multiply(batch, vector)
    assert torch.allclose(
        selected.inner(batch, vector, right),
        vector["w"] @ (DenseMetricData.matrix @ right["w"]),
    )


def metric_materializer_thresholds() -> dict[str, float]:
    return {
        "max_abs_diff": 1e-12,
        "max_rel_diff": 1e-12,
        "symmetry_max_abs_diff": 1e-12,
        "psd_violation": 1e-12,
    }


def inverse_metric_materializer_thresholds(
    condition_number_max: float,
) -> dict[str, float]:
    return {
        **metric_materializer_thresholds(),
        "inverse_residual": 1e-12,
        "damping_min": 0.1,
        "condition_number_max": condition_number_max,
    }


def materialized_standard_metric_operator(
    tmp_path: Path,
    *,
    model: torch.nn.Module,
    data: vpx.DataProvider,
    operator: vpx.OperatorSpec,
    vectors: vpx.VectorProvider,
    candidate: vpx.Candidate,
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, str],
    family: str,
) -> vpx.StandardMetricOperator:
    params = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=data,
        operator=operator,
        vectors=vectors,
        target=cpu_target(),
        runtime=vpx.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=(candidate,),
            thresholds=thresholds,
            objective_signature=objective_signature,
            axis_registry=vpx.standard_axis_registry(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, name=family)
    assert isinstance(selected, vpx.StandardMetricOperator)

    return selected


def low_rank_metric_matrix() -> torch.Tensor:
    factors = LowRankMetricData.factors()
    basis = factors["basis"]
    diagonal = factors["diagonal"]

    return basis @ basis.T + torch.diag(diagonal)


def kfac_metric_matrix() -> torch.Tensor:
    factors = KFACMetricData.factors()

    return torch.kron(factors["w_left"], factors["w_right"])


def ggn_metric_matrix() -> torch.Tensor:
    factors = GGNMetricData.factors()
    jacobian = factors["jacobian"]
    loss_hessian = factors["loss_hessian"]

    return jacobian.T @ loss_hessian @ jacobian


@pytest.mark.parametrize(
    (
        "case_id",
        "model",
        "data",
        "operator",
        "vectors",
        "candidate",
        "objective_signature",
        "batch",
        "vector",
        "right",
        "expected",
        "expected_inner",
    ),
    [
        (
            "diagonal",
            TwoParameterModule(),
            DiagonalMetricData(),
            ops.metric(
                "metric",
                "diagonal",
                aggregation="sum",
                representation=diagonal_metric_representation(),
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "metric",
                "factorized",
                metric_settings("factorized_multiply"),
                changed_axes=("metric.multiply_path",),
            ),
            {"diagonal": "metric-v1"},
            {"metric_diagonal": DiagonalMetricData.metric_diagonal()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            {"w": torch.tensor([-1.0, 0.5], dtype=torch.float64)},
            torch.tensor([4.0, 10.0], dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
        (
            "blocks",
            TwoParameterModule(),
            BlockMetricData(),
            ops.metric(
                "metric",
                "blocks",
                aggregation="sum",
                representation=block_metric_representation(),
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "metric",
                "blockwise",
                metric_settings("blockwise_multiply"),
                changed_axes=("metric.multiply_path",),
            ),
            {"blocks": "metric-v1"},
            {"metric_blocks": BlockMetricData.metric_blocks()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            {"w": torch.tensor([-1.0, 0.5], dtype=torch.float64)},
            torch.tensor([6.0, 7.0], dtype=torch.float64),
            torch.tensor(-2.5, dtype=torch.float64),
        ),
        (
            "low_rank",
            TwoParameterModule(),
            LowRankMetricData(),
            ops.metric(
                "metric",
                "low_rank",
                aggregation="sum",
                representation=low_rank_metric_representation(),
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "metric",
                "factorized",
                metric_settings("factorized_multiply"),
                changed_axes=("metric.multiply_path",),
            ),
            {"low_rank": "metric-v1"},
            {"low_rank_factors": LowRankMetricData.factors()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            {"w": torch.tensor([-1.0, 0.5], dtype=torch.float64)},
            low_rank_metric_matrix() @ torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
        (
            "kfac",
            MatrixParameterModule(),
            KFACMetricData(),
            ops.metric(
                "metric",
                "kfac",
                aggregation="sum",
                representation=kfac_metric_representation(),
            ),
            MatrixVectorProvider(),
            vpx.Candidate(
                "metric",
                "factorized",
                metric_settings("factorized_multiply"),
                changed_axes=("metric.multiply_path",),
            ),
            {"kfac": "metric-v1"},
            {"kfac_factors": KFACMetricData.factors()},
            {"w": matrix_vector_tensor()},
            None,
            kfac_metric_matrix() @ matrix_vector_tensor().reshape(-1),
            None,
        ),
        (
            "ggn",
            TwoParameterModule(),
            GGNMetricData(),
            ops.metric(
                "metric",
                "ggn",
                aggregation="sum",
                representation=ggn_metric_representation(),
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "metric",
                "factorized",
                metric_settings("factorized_multiply"),
                changed_axes=("metric.multiply_path",),
            ),
            {"ggn": "metric-v1"},
            {"ggn_factors": GGNMetricData.factors()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            None,
            ggn_metric_matrix() @ torch.tensor([1.0, 2.0], dtype=torch.float64),
            None,
        ),
    ],
)
def test_standard_metric_materializer_preserves_representation(
    tmp_path: Path,
    case_id: str,
    model: torch.nn.Module,
    data: vpx.DataProvider,
    operator: vpx.OperatorSpec,
    vectors: vpx.VectorProvider,
    candidate: vpx.Candidate,
    objective_signature: Mapping[str, str],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    right: vpx.TensorTree | None,
    expected: torch.Tensor,
    expected_inner: torch.Tensor | None,
) -> None:
    assert case_id
    selected = materialized_standard_metric_operator(
        tmp_path,
        model=model,
        data=data,
        operator=operator,
        vectors=vectors,
        candidate=candidate,
        thresholds=metric_materializer_thresholds(),
        objective_signature=objective_signature,
        family="metric",
    )

    torch.testing.assert_close(flatten_tree(selected(batch, vector)), expected)

    if right is not None:
        torch.testing.assert_close(selected.inner(batch, vector, right), expected_inner)


@pytest.mark.parametrize(
    (
        "case_id",
        "model",
        "data",
        "operator",
        "vectors",
        "candidate",
        "objective_signature",
        "condition_number_max",
        "batch",
        "vector",
        "expected",
    ),
    [
        (
            "low_rank",
            TwoParameterModule(),
            LowRankMetricData(),
            ops.inverse_metric(
                "inverse_metric",
                "low_rank",
                aggregation="sum",
                representation=low_rank_metric_representation(),
                damping=0.25,
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "inverse_metric",
                "woodbury",
                inverse_metric_settings("woodbury_low_rank_solve"),
                changed_axes=("inverse_metric.solve_path",),
            ),
            {"low_rank": "inverse-metric-v1"},
            5.0,
            {"low_rank_factors": LowRankMetricData.factors()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            torch.linalg.solve(
                low_rank_metric_matrix() + 0.25 * torch.eye(2, dtype=torch.float64),
                torch.tensor([1.0, 2.0], dtype=torch.float64),
            ),
        ),
        (
            "kfac",
            MatrixParameterModule(),
            KFACMetricData(),
            ops.inverse_metric(
                "inverse_metric",
                "kfac",
                aggregation="sum",
                representation=kfac_metric_representation(),
                damping=0.25,
            ),
            MatrixVectorProvider(),
            vpx.Candidate(
                "inverse_metric",
                "factorized",
                inverse_metric_settings("factorized_solve"),
                changed_axes=("inverse_metric.solve_path",),
            ),
            {"kfac": "inverse-metric-v1"},
            20.0,
            {"kfac_factors": KFACMetricData.factors()},
            {"w": matrix_vector_tensor()},
            torch.linalg.solve(
                kfac_metric_matrix() + 0.25 * torch.eye(4, dtype=torch.float64),
                matrix_vector_tensor().reshape(-1),
            ),
        ),
        (
            "ggn",
            TwoParameterModule(),
            GGNMetricData(),
            ops.inverse_metric(
                "inverse_metric",
                "ggn",
                aggregation="sum",
                representation=ggn_metric_representation(),
                damping=0.25,
            ),
            TwoParameterVectorProvider(),
            vpx.Candidate(
                "inverse_metric",
                "factorized",
                inverse_metric_settings("factorized_solve"),
                changed_axes=("inverse_metric.solve_path",),
            ),
            {"ggn": "inverse-metric-v1"},
            10.0,
            {"ggn_factors": GGNMetricData.factors()},
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            torch.linalg.solve(
                ggn_metric_matrix() + 0.25 * torch.eye(2, dtype=torch.float64),
                torch.tensor([1.0, 2.0], dtype=torch.float64),
            ),
        ),
    ],
)
def test_standard_inverse_metric_materializer_preserves_representation(
    tmp_path: Path,
    case_id: str,
    model: torch.nn.Module,
    data: vpx.DataProvider,
    operator: vpx.OperatorSpec,
    vectors: vpx.VectorProvider,
    candidate: vpx.Candidate,
    objective_signature: Mapping[str, str],
    condition_number_max: float,
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    expected: torch.Tensor,
) -> None:
    assert case_id
    selected = materialized_standard_metric_operator(
        tmp_path,
        model=model,
        data=data,
        operator=operator,
        vectors=vectors,
        candidate=candidate,
        thresholds=inverse_metric_materializer_thresholds(condition_number_max),
        objective_signature=objective_signature,
        family="inverse_metric",
    )

    torch.testing.assert_close(flatten_tree(selected(batch, vector)), expected)


def test_runtime_config_propagates_recomputed_teacher_objective(tmp_path: Path) -> None:
    model = OneParameterModule()
    calls = []

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        teacher = batch["teacher_outputs"]
        assert isinstance(teacher, Mapping)
        logits = teacher["logits"]
        assert isinstance(logits, torch.Tensor)

        return batch["scale"] * params["w"].pow(2).sum() + logits.sum() * 0.0

    operator = ops.gradient("gradient", "loss", aggregation="sum")
    settings = {
        **gradient_settings(),
        "teacher_outputs": "recomputed_with_equality_check",
    }
    candidate = vpx.Candidate(
        "gradient",
        "teacher-recompute",
        settings,
        changed_axes=tuple(settings),
    )
    thresholds = {
        "max_abs_diff": 1e-6,
        "max_rel_diff": 1e-6,
        "directional_abs_diff": 1e-3,
        "directional_rel_diff": 1e-3,
    }
    runtime = vpx.standard_runtime_config(
        operator,
        params={"w": model.w},
        buffers={},
        candidates=(candidate,),
        thresholds=thresholds,
        objective_signature={"loss": "quadratic-with-teacher-v1"},
        axis_registry=vpx.standard_axis_registry(),
        parameter_surface=vpx.parameter_surface(model),
        scalar_objectives={"loss": scalar},
        teacher_objective=RecordingTeacherObjective("teacher-v1", calls),
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=TeacherOutputData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
        anchor_policy={},
        replay_policy={},
        adapter_identity={"adapter_id": "test", "adapter_version": "1"},
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected_candidate().candidate_id == "teacher-recompute"
    assert calls == ["gradient", "gradient"]


def test_standard_runtime_identity_records_callback_identity() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    candidate = vpx.Candidate(
        "gradient",
        "teacher-recompute",
        {**gradient_settings(), "teacher_outputs": "recomputed_with_equality_check"},
        admission_status="passed",
    )
    first = vpx.standard_runtime_config(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        candidates=(candidate,),
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "directional_abs_diff": 1e-12,
            "directional_rel_diff": 1e-12,
        },
        objective_signature={"loss": "identity-test"},
        axis_registry=vpx.standard_axis_registry(),
        scalar_objectives={"loss": quadratic_scalar},
        teacher_objective=IdentifiedTeacherObjective("first"),
    )
    second = vpx.standard_runtime_config(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        candidates=(candidate,),
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "directional_abs_diff": 1e-12,
            "directional_rel_diff": 1e-12,
        },
        objective_signature={"loss": "identity-test"},
        axis_registry=vpx.standard_axis_registry(),
        scalar_objectives={"loss": quadratic_scalar},
        teacher_objective=IdentifiedTeacherObjective("second"),
    )

    assert first.identity()["teacher_objective"] == {
        "kind": "explicit_identity",
        "value": {"version": "first"},
    }
    assert first.identity() != second.identity()


def test_standard_runtime_identity_rejects_closure_callbacks() -> None:
    calls = []

    def teacher_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert params
        assert buffers == {}
        assert batch
        assert context.family == "gradient"
        calls.append(context.candidate_id)

        return {"logits": torch.tensor([1.0], dtype=torch.float64)}

    with pytest.raises(vp.MaterializationError, match="closes over runtime state"):
        vpx.standard_runtime_config(
            ops.gradient("gradient", "loss", aggregation="sum"),
            params={"w": torch.tensor([2.0], dtype=torch.float64)},
            buffers={},
            candidates=(
                vpx.Candidate(
                    "gradient",
                    "teacher-recompute",
                    {
                        **gradient_settings(),
                        "teacher_outputs": "recomputed_with_equality_check",
                    },
                    admission_status="passed",
                ),
            ),
            thresholds={
                "max_abs_diff": 1e-12,
                "max_rel_diff": 1e-12,
                "directional_abs_diff": 1e-12,
                "directional_rel_diff": 1e-12,
            },
            objective_signature={"loss": "identity-test"},
            axis_registry=vpx.standard_axis_registry(),
            scalar_objectives={"loss": quadratic_scalar},
            teacher_objective=teacher_objective,
        )


def test_inverse_metric_materializer_calls_inverse_by_default(tmp_path: Path) -> None:
    model = TwoParameterModule()
    params = {"w": model.w.detach().clone()}
    operator = ops.inverse_metric(
        "inverse_metric",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.25,
    )
    candidates = (
        vpx.Candidate(
            "inverse_metric",
            "cholesky",
            inverse_metric_settings("cholesky_solve"),
            changed_axes=("inverse_metric.solve_path",),
        ),
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=DenseMetricData(),
        operator=operator,
        vectors=TwoParameterVectorProvider(),
        target=cpu_target(),
        runtime=vpx.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=candidates,
            thresholds={
                "max_abs_diff": 1e-12,
                "max_rel_diff": 1e-12,
                "symmetry_max_abs_diff": 1e-12,
                "psd_violation": 1e-12,
                "inverse_residual": 1e-12,
                "damping_min": 0.1,
                "condition_number_max": 10.0,
            },
            objective_signature={"dense": "inverse-metric-v1"},
            axis_registry=vpx.standard_axis_registry(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, name="inverse_metric")
    batch = {"metric_matrix": DenseMetricData.matrix}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    assert isinstance(selected, vpx.StandardMetricOperator)
    assert torch.allclose(
        tree_leaves(selected(batch, vector))[0],
        torch.linalg.solve(
            DenseMetricData.matrix + 0.25 * torch.eye(2, dtype=torch.float64),
            vector["w"],
        ),
    )
    assert torch.allclose(
        tree_leaves(selected.inverse_multiply(batch, vector))[0],
        torch.linalg.solve(
            DenseMetricData.matrix + 0.25 * torch.eye(2, dtype=torch.float64),
            vector["w"],
        ),
    )


def test_inverse_metric_materializer_preserves_conjugate_gradient_path(
    tmp_path: Path,
) -> None:
    model = TwoParameterModule()
    params = {"w": model.w.detach().clone()}
    operator = ops.inverse_metric(
        "inverse_metric",
        "dense",
        aggregation="sum",
        representation=dense_metric_representation(),
        damping=0.25,
    )
    candidates = (
        vpx.Candidate(
            "inverse_metric",
            "cg",
            {
                "inverse_metric.solve_path": "conjugate_gradient",
                "inverse_metric.iteration_budget": 2,
                "inverse_metric.preconditioner": "none",
                "metric.multiply_path": "dense_matmul",
            },
            changed_axes=("inverse_metric.solve_path",),
        ),
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=DenseMetricData(),
        operator=operator,
        vectors=TwoParameterVectorProvider(),
        target=cpu_target(),
        runtime=vpx.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=candidates,
            thresholds={
                "max_abs_diff": 1e-12,
                "max_rel_diff": 1e-12,
                "symmetry_max_abs_diff": 1e-12,
                "psd_violation": 1e-12,
                "inverse_residual": 1e-12,
                "damping_min": 0.1,
                "condition_number_max": 10.0,
            },
            objective_signature={"dense": "inverse-metric-v1"},
            axis_registry=vpx.standard_axis_registry(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, name="inverse_metric")
    batch = {"metric_matrix": DenseMetricData.matrix}
    vector = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    assert isinstance(selected, vpx.StandardMetricOperator)
    assert torch.allclose(
        tree_leaves(selected(batch, vector))[0],
        torch.linalg.solve(
            DenseMetricData.matrix + 0.25 * torch.eye(2, dtype=torch.float64),
            vector["w"],
        ),
    )


def flatten_tree(tree: vpx.TensorTree) -> torch.Tensor:
    return torch.cat(tuple(value.reshape(-1) for value in tree_leaves(tree)))


def tensor_mapping(tree: object) -> dict[str, torch.Tensor]:
    assert isinstance(tree, Mapping)
    values = {}

    for key, value in tree.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)
        values[key] = value

    return values


def assert_two_leaf_ordered_map(tree: object, expected_flat: torch.Tensor) -> None:
    tensor_map = tensor_mapping(tree)

    assert tuple(tensor_map) == ("a", "b")
    assert torch.allclose(tensor_map["a"], expected_flat[:2])
    assert torch.allclose(tensor_map["b"], expected_flat[2:].reshape(2, 1))


def assert_two_leaf_batched_map(tree: object, expected_flat: torch.Tensor) -> None:
    tensor_map = tensor_mapping(tree)

    assert tuple(tensor_map) == ("a", "b")
    torch.testing.assert_close(tensor_map["a"], expected_flat[:, :2])
    torch.testing.assert_close(
        tensor_map["b"],
        expected_flat[:, 2:].reshape(expected_flat.shape[0], 2, 1),
    )


def standard_operation_result(
    operator: Any,
    params: vpx.ParameterTree,
    batch: Mapping[str, Any],
    vector: vpx.TensorTree,
    settings: Mapping[str, Any],
    candidate_id: str,
) -> vpx.TensorTree:
    factory = vpx.standard_operation_factory(operator, params=params, buffers={})

    return factory(
        passed_candidate(
            operator.family,
            candidate_id,
            settings,
        ),
        batch,
        vector,
    )()


def repeated_standard_operation_maps(
    operator: Any,
    params: vpx.ParameterTree,
    batch: Mapping[str, Any],
    vector: vpx.TensorTree,
    settings: Mapping[str, Any],
    candidate_id: str,
    *,
    scalar_objectives: Mapping[str, Any] | None = None,
    function_objectives: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )
    operation = factory(
        passed_candidate(
            operator.family,
            candidate_id,
            settings,
        ),
        batch,
        vector,
    )

    return tensor_mapping(operation()), tensor_mapping(operation())


def metric_inner_factored_gram_result(
    operator: Any,
    params: vpx.ParameterTree,
    batch: Mapping[str, Any],
    left: vpx.TensorTree,
    right: vpx.TensorTree,
) -> torch.Tensor:
    family = operator.kind
    result = standard_operation_result(
        operator,
        params,
        batch,
        (left, right),
        {
            f"{family}.reduction_path": "factored_gram",
            f"{family}.multi_rhs": "single_column",
        },
        "factored-gram",
    )
    assert isinstance(result, torch.Tensor)

    return result


def metric_inner_single_column_pair_results(
    metric_operator: Any,
    inverse_operator: Any,
    params: vpx.ParameterTree,
    batch: Mapping[str, Any],
    left: vpx.TensorTree,
    right: vpx.TensorTree,
    metric_reduction_path: str,
    inverse_reduction_path: str,
    shared_settings: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    metric_result = standard_operation_result(
        metric_operator,
        params,
        batch,
        (left, right),
        {
            f"{metric_operator.kind}.reduction_path": metric_reduction_path,
            f"{metric_operator.kind}.multi_rhs": "single_column",
            **shared_settings,
        },
        "metric-inner",
    )
    inverse_result = standard_operation_result(
        inverse_operator,
        params,
        batch,
        (left, right),
        {
            f"{inverse_operator.kind}.reduction_path": inverse_reduction_path,
            f"{inverse_operator.kind}.multi_rhs": "single_column",
            **shared_settings,
        },
        "inverse-metric-inner",
    )
    assert isinstance(metric_result, torch.Tensor)
    assert isinstance(inverse_result, torch.Tensor)

    return metric_result, inverse_result


def metric_square_root_pair_results(
    sqrt_operator: Any,
    inverse_operator: Any,
    params: vpx.ParameterTree,
    batch: Mapping[str, Any],
    settings: Mapping[str, Any],
    sqrt_vector: vpx.TensorTree,
    inverse_vector: vpx.TensorTree,
) -> tuple[vpx.TensorTree, vpx.TensorTree]:
    sqrt_result = standard_operation_result(
        sqrt_operator,
        params,
        batch,
        sqrt_vector,
        settings,
        "sqrt",
    )
    inverse_result = standard_operation_result(
        inverse_operator,
        params,
        batch,
        inverse_vector,
        settings,
        "inverse-sqrt",
    )

    return sqrt_result, inverse_result


def test_kfac_metric_operator_matches_dense_reference_for_single_block() -> None:
    left = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    right = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    dense = torch.kron(left, right)
    batch = {"a": left, "b": right}
    vector = {"w": torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)}
    other = {"w": torch.tensor([[0.25, -0.75], [1.5, 2.0]], dtype=torch.float64)}
    metric = vpx.KFACMetricOperator((vpx.KFACMetricBlock("w", "a", "b"),))

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
    metric = vpx.KFACMetricOperator((
        vpx.KFACMetricBlock("w", "w_left", "w_right"),
        vpx.KFACMetricBlock("b", "b_left", "b_right"),
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
    mean_factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    sum_factory = vpx.standard_operation_factory(
        ops.empirical_fisher_vp(
            "empirical",
            "scores",
            aggregation="sum",
            example_loss_reduction="per_example",
            denominator="batch_normalization",
        ),
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError):
        mean_factory(
            vpx.Candidate(
                "fisher",
                "row",
                fisher_settings("materialize_score_gradients"),
                admission_status="passed",
            ),
            {"score_gradients": score_gradients, "normalization": 0.0},
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="score_gradients"):
        mean_factory(
            vpx.Candidate(
                "fisher",
                "row",
                fisher_settings("materialize_score_gradients"),
                admission_status="passed",
            ),
            {"normalization": 1.0},
            vector,
        )

    with pytest.raises(vp.MaterializationError):
        sum_factory(
            vpx.Candidate(
                "empirical",
                "row",
                empirical_dense_settings(),
                admission_status="passed",
            ),
            {"per_example_gradients": score_gradients, "normalization": 2.0},
            vector,
        )()


@pytest.mark.parametrize(
    ("operator", "family", "settings", "batch_key", "batch_value", "extra_batch"),
    [
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            fisher_settings("materialize_score_gradients"),
            "score_gradients",
            torch.empty((0, 2), dtype=torch.float64),
            {"normalization": 1.0},
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            sampled_fisher_settings("materialize_score_gradients"),
            "sampled_score_gradients",
            torch.empty((0, 2), dtype=torch.float64),
            {"num_examples": 1},
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="batch_normalization",
            ),
            "empirical",
            empirical_dense_settings(),
            "per_example_gradients",
            torch.empty((0, 2), dtype=torch.float64),
            {"normalization": 1.0},
        ),
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            fisher_settings("blockwise_score_matrix"),
            "score_gradient_blocks",
            (
                torch.empty((0, 1), dtype=torch.float64),
                torch.empty((0, 1), dtype=torch.float64),
            ),
            {"normalization": 1.0},
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            sampled_fisher_settings("blockwise_score_matrix"),
            "sampled_score_gradient_blocks",
            (
                torch.empty((0, 1), dtype=torch.float64),
                torch.empty((0, 1), dtype=torch.float64),
            ),
            {"num_examples": 1},
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="batch_normalization",
            ),
            "empirical",
            {"empirical_fisher.accumulation": "blockwise_gradient_matrix"},
            "per_example_gradient_blocks",
            (
                torch.empty((0, 1), dtype=torch.float64),
                torch.empty((0, 1), dtype=torch.float64),
            ),
            {"normalization": 1.0},
        ),
    ],
)
def test_fisher_style_runtime_rejects_empty_score_surface(
    operator: vpx.OperatorSpec,
    family: str,
    settings: Mapping[str, object],
    batch_key: str,
    batch_value: object,
    extra_batch: Mapping[str, object],
) -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    vector = {"w": torch.tensor([0.4, -0.7], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="at least one row"):
        factory(
            vpx.Candidate(
                family,
                "empty-score-surface",
                settings,
                admission_status="passed",
            ),
            {
                batch_key: batch_value,
                **dict(extra_batch),
            },
            vector,
        )()


@pytest.mark.parametrize(
    ("operator", "family", "settings", "batch_key", "batch_value", "extra_batch"),
    [
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            fisher_settings("materialize_score_gradients"),
            "score_gradients",
            _float_rows(FISHER_ZERO_SCORE_ROWS),
            {"normalization": 1.0},
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            sampled_fisher_settings("materialize_score_gradients"),
            "sampled_score_gradients",
            _float_rows(FISHER_ZERO_SCORE_ROWS),
            {"num_examples": 1},
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="batch_normalization",
            ),
            "empirical",
            empirical_dense_settings(),
            "per_example_gradients",
            _float_rows(FISHER_ZERO_SCORE_ROWS),
            {"normalization": 1.0},
        ),
        (
            score_terms_fisher("fisher", "scores"),
            "fisher",
            fisher_settings("blockwise_score_matrix"),
            "score_gradient_blocks",
            COLUMN_BLOCKS(_float_rows(FISHER_ZERO_SCORE_ROWS)),
            {"normalization": 1.0},
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            "sampled",
            sampled_fisher_settings("blockwise_score_matrix"),
            "sampled_score_gradient_blocks",
            COLUMN_BLOCKS(_float_rows(FISHER_ZERO_SCORE_ROWS)),
            {"num_examples": 1},
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="batch_normalization",
            ),
            "empirical",
            {"empirical_fisher.accumulation": "blockwise_gradient_matrix"},
            "per_example_gradient_blocks",
            COLUMN_BLOCKS(_float_rows(FISHER_ZERO_SCORE_ROWS)),
            {"normalization": 1.0},
        ),
    ],
)
def test_fisher_style_runtime_maps_zero_vector_to_zero(
    operator: vpx.OperatorSpec,
    family: str,
    settings: Mapping[str, object],
    batch_key: str,
    batch_value: object,
    extra_batch: Mapping[str, object],
) -> None:
    params = {"w": torch.tensor([0.3, -0.2], dtype=torch.float64)}
    vector = {"w": torch.zeros(2, dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    result = factory(
        vpx.Candidate(
            family,
            "zero-vector",
            settings,
            admission_status="passed",
        ),
        {
            batch_key: batch_value,
            **dict(extra_batch),
        },
        vector,
    )()

    for leaf in tree_leaves(result):
        torch.testing.assert_close(leaf, torch.zeros_like(leaf))


def test_dense_metric_uses_vector_order_and_fisher_uses_parameter_order() -> None:
    params = {
        "a": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "b": torch.tensor([[0.0], [0.0]], dtype=torch.float64),
    }
    vector = {
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }
    matrix = torch.arange(16.0, dtype=torch.float64).reshape(4, 4)
    score_gradients = torch.eye(4, dtype=torch.float64)
    metric_factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers={},
    )
    fisher_factory = vpx.standard_operation_factory(
        score_terms_fisher("fisher", "scores"),
        params=params,
        buffers={},
    )
    empirical_factory = vpx.standard_operation_factory(
        empirical_fisher_sum("empirical", "scores"),
        params=params,
        buffers={},
    )
    metric_result = run_passed_candidate(
        metric_factory,
        "metric",
        "row",
        metric_settings(),
        {"metric_matrix": matrix},
        vector,
    )
    fisher_result = run_passed_candidate(
        fisher_factory,
        "fisher",
        "row",
        fisher_settings("materialize_score_gradients"),
        {"score_gradients": score_gradients, "normalization": 1.0},
        vector,
    )
    empirical_result = run_passed_candidate(
        empirical_factory,
        "empirical",
        "row",
        empirical_dense_settings(),
        {"per_example_gradients": score_gradients},
        vector,
    )
    flat_vector = torch.tensor([3.0, 4.0, 1.0, 2.0], dtype=torch.float64)
    metric_flat = matrix @ flat_vector
    metric_leaves = tree_leaves(metric_result)
    expected_parameter_order = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )

    assert torch.allclose(metric_leaves[0], metric_flat[:2].reshape(2, 1))
    assert torch.allclose(metric_leaves[1], metric_flat[2:])
    assert_two_leaf_ordered_map(fisher_result, expected_parameter_order)
    assert_two_leaf_ordered_map(empirical_result, expected_parameter_order)


def test_sampled_fisher_vp_dense_loop_and_anchor_use_parameter_order() -> None:
    params = {
        "a": torch.zeros(2, dtype=torch.float64),
        "b": torch.zeros((2, 1), dtype=torch.float64),
    }
    vector = {
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }
    sampled_score_gradients = torch.eye(4, dtype=torch.float64)
    batch = {
        "sampled_score_gradients": sampled_score_gradients,
        "num_examples": 2,
    }

    def sampled_scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["num_examples"] == 2
        assert context.family == "sampled"

        return torch.stack((
            params["a"][0],
            params["a"][1],
            params["b"][0, 0],
            params["b"][1, 0],
        ))

    operator = score_terms_sampled_fisher("sampled", "sampled_scores")
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        function_objectives={"sampled_scores": sampled_scores},
    )
    results = [
        run_passed_candidate(factory, "sampled", candidate_id, settings, batch, vector)
        for candidate_id, settings in (
            ("dense", sampled_fisher_settings("materialize_score_gradients")),
            (
                "loop",
                {
                    **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                    "schedule.per_example": "loop",
                },
            ),
        )
    ]
    check = vpx.standard_reference_check(
        operator,
        params=params,
        buffers={},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
        function_objectives={"sampled_scores": sampled_scores},
    )
    reference_result = check(
        passed_candidate(
            "sampled",
            "dense-reference",
            sampled_fisher_settings("materialize_score_gradients"),
        ),
        batch,
        vector,
    )
    expected_flat = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

    for result in results:
        assert_two_leaf_ordered_map(result, expected_flat)

    assert reference_result.measurements["max_abs_diff"] == pytest.approx(0.0)

    bounded_result = run_passed_candidate(
        factory,
        "sampled",
        "bounded",
        sampled_fisher_settings(
            "materialize_score_gradients",
            exact_check="enabled_with_sampling_bound",
        ),
        {
            **batch,
            "exact_fisher_vp": expected_flat,
        },
        vector,
    )

    assert_two_leaf_ordered_map(bounded_result, expected_flat)

    with pytest.raises(vp.MaterializationError, match="exceeded bound"):
        run_passed_candidate(
            factory,
            "sampled",
            "bounded-failed",
            sampled_fisher_settings(
                "materialize_score_gradients",
                exact_check="enabled_with_sampling_bound",
            ),
            {
                **batch,
                "exact_fisher_vp": expected_flat + 1.0,
            },
            vector,
        )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            passed_candidate(
                "sampled",
                "wrong-dense-reference",
                sampled_fisher_settings("materialize_score_gradients"),
            ),
            {
                "sampled_score_gradients": torch.zeros_like(sampled_score_gradients),
                "num_examples": 2,
            },
            vector,
        )


@pytest.mark.parametrize(
    "sampling_bound",
    [
        {
            "kind": "matrix_bernstein",
            "failure_probability": 0.5,
            "norm_floor": 1e-12,
        },
        {
            "kind": "hutchinson_relative_variance",
            "failure_probability": 0.5,
            "norm_floor": 1e-12,
        },
    ],
)
def test_sampled_fisher_named_exact_bounds_execute(
    sampling_bound: Mapping[str, object],
) -> None:
    params = {
        "a": torch.zeros(2, dtype=torch.float64),
        "b": torch.zeros((2, 1), dtype=torch.float64),
    }
    vector = {
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }
    sampled_score_gradients = torch.eye(4, dtype=torch.float64)
    expected_flat = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    operator = score_terms_sampled_fisher(
        "sampled",
        "sampled_scores",
        sampling_bound=sampling_bound,
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )
    result = run_passed_candidate(
        factory,
        "sampled",
        "named-bound",
        sampled_fisher_settings(
            "materialize_score_gradients",
            exact_check="enabled_with_sampling_bound",
        ),
        {
            "sampled_score_gradients": sampled_score_gradients,
            "num_examples": 2,
            "exact_fisher_vp": expected_flat,
        },
        vector,
    )

    assert_two_leaf_ordered_map(result, expected_flat)


def test_sampled_fisher_vp_score_grad_paths_match_loop_path() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {
        "coeff": torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64),
        "num_examples": 2,
    }

    def sampled_scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "sampled"
        coeff = batch["coeff"]
        assert isinstance(coeff, torch.Tensor)

        return params["w"][0] * coeff.reshape(-1)

    factory = vpx.standard_operation_factory(
        score_terms_sampled_fisher("sampled", "sampled_scores"),
        params=params,
        buffers={},
        function_objectives={"sampled_scores": sampled_scores},
    )
    results = candidate_leaf_results(
        factory,
        "sampled",
        (
            ("loop", sampled_fisher_grad_settings("torch_autograd_grad_loop")),
            (
                "torch-func",
                {
                    **sampled_fisher_grad_settings("torch_func_grad"),
                    "schedule.per_example": "loop",
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
            (
                "backward",
                {
                    **sampled_fisher_grad_settings("backward_materialized_grad"),
                    "schedule.per_example": "loop",
                },
            ),
            (
                "vmap",
                {
                    **sampled_fisher_grad_settings("vmap_grad"),
                    **fisher_per_example_vmap_settings(),
                    **torch_func_settings(requires_forward_ad=False),
                },
            ),
        ),
        batch,
        vector,
    )

    assert torch.allclose(
        results["loop"],
        torch.tensor([90.0], dtype=torch.float64),
    )

    for candidate_id in ("torch-func", "backward", "vmap"):
        assert torch.allclose(results[candidate_id], results["loop"])


def test_sampled_fisher_vp_rejects_inconsistent_rows() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {
        "sampled_score_gradients": torch.ones((4, 1), dtype=torch.float64),
        "num_examples": 2,
    }

    def sampled_scores(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["num_examples"] == 2
        assert context.family == "sampled"

        return params["w"].repeat(4)

    operator = score_terms_sampled_fisher("sampled", "sampled_scores")
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
        function_objectives={"sampled_scores": sampled_scores},
    )

    with pytest.raises(vp.MaterializationError, match=r"sampled_fisher\.accumulation"):
        factory(
            vpx.Candidate(
                "sampled",
                "missing-accumulation",
                {
                    "sampled_fisher.sample_source": "fixed_seed_and_count",
                    "sampled_fisher.exact_fisher_check": "disabled",
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )

    with pytest.raises(vp.MaterializationError, match="score_grad_path"):
        factory(
            vpx.Candidate(
                "sampled",
                "missing-score-path",
                sampled_fisher_settings("streaming_dot_accumulate"),
                admission_status="passed",
            ),
            batch,
            vector,
        )

    with pytest.raises(vp.MaterializationError, match="not used"):
        factory(
            vpx.Candidate(
                "sampled",
                "dense-with-score-path",
                {
                    **sampled_fisher_settings("materialize_score_gradients"),
                    "sampled_fisher.score_grad_path": "torch_autograd_grad_loop",
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )

    with pytest.raises(vp.MaterializationError, match="sample_source"):
        factory(
            vpx.Candidate(
                "sampled",
                "missing-source",
                {
                    "sampled_fisher.accumulation": "materialize_score_gradients",
                    "sampled_fisher.exact_fisher_check": "disabled",
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="differs from operator"):
        factory(
            vpx.Candidate(
                "sampled",
                "different-source",
                {
                    **sampled_fisher_settings("materialize_score_gradients"),
                    "sampled_fisher.sample_source": "fixed_sample_table",
                },
                admission_status="passed",
            ),
            batch,
            vector,
        )()

    with pytest.raises(vp.MaterializationError, match="exact_fisher_vp"):
        factory(
            vpx.Candidate(
                "sampled",
                "exact-check",
                sampled_fisher_settings(
                    "materialize_score_gradients",
                    exact_check="enabled_with_sampling_bound",
                ),
                admission_status="passed",
            ),
            batch,
            vector,
        )()


def test_sampled_fisher_exact_check_requires_declared_sampling_bound() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    batch = {
        "sampled_score_gradients": torch.ones((4, 1), dtype=torch.float64),
        "num_examples": 2,
    }
    operator = ops.sampled_fisher_vp(
        "sampled",
        "sampled_scores",
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="sampled_labels",
        sample_count=2,
        sample_source="fixed_seed_and_count",
        sampling_bound={"kind": "disabled"},
        score_reduction="none",
        denominator="num_examples",
    )
    factory = vpx.standard_operation_factory(
        operator,
        params=params,
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="declared sampling_bound"):
        factory(
            vpx.Candidate(
                "sampled",
                "exact-check-without-bound",
                sampled_fisher_settings(
                    "materialize_score_gradients",
                    exact_check="enabled_with_sampling_bound",
                ),
                admission_status="passed",
            ),
            batch,
            vector,
        )()


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
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch["loss_hessian"] is loss_hessian
        assert context.family == "ggn"

        return torch.stack((
            params["a"][0] + params["b"][0, 0],
            params["a"][1] - params["b"][1, 0],
        ))

    factory = vpx.standard_operation_factory(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params=params,
        buffers={},
        function_objectives={"model_output": function},
    )
    result = factory(
        vpx.Candidate(
            "ggn",
            "row",
            ggn_dense_kernel_settings(),
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
    result_map = tensor_mapping(result)
    result_a = result_map["a"]
    result_b = result_map["b"]

    assert torch.allclose(result_a, expected[:2])
    assert torch.allclose(result_b, expected[2:].reshape(2, 1))


def test_hvp_vhp_path_supports_parameter_tree_order() -> None:
    params = {
        "a": torch.tensor([0.5, -0.25], dtype=torch.float64),
        "b": torch.tensor([[0.1], [0.2]], dtype=torch.float64),
    }
    vector = {
        "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert isinstance(batch["scale"], float)
        assert context.family == "hvp"

        return batch["scale"] * (params["a"].pow(2).sum() + params["b"].pow(2).sum())

    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "hvp",
            "row",
            hvp_settings("autograd_functional_vhp"),
            admission_status="passed",
        ),
        {"scale": 5.0},
        vector,
    )()
    result_map = tensor_mapping(result)
    result_a = result_map["a"]
    result_b = result_map["b"]

    assert torch.allclose(result_a, 10.0 * vector["a"])
    assert torch.allclose(result_b, 10.0 * vector["b"])


def test_standard_runtime_executes_dtype_and_backend_axes() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    matrix = torch.eye(1, dtype=torch.float64)
    observed = {}
    metric_factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params=params,
        buffers=buffers,
    )

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert context.family == "gradient"
        observed["param_dtype"] = params["w"].dtype
        observed["buffer_dtype"] = buffers["b"].dtype
        observed["floating_dtype"] = batch["floating"].dtype
        observed["input_ids_dtype"] = batch["input_ids"].dtype
        observed["attention_mask_dtype"] = batch["attention_mask"].dtype
        observed["matmul_precision"] = torch.get_float32_matmul_precision()
        observed["allow_bf16"] = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
        observed["allow_fp16"] = (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        )
        observed["deterministic"] = torch.are_deterministic_algorithms_enabled()

        return batch["floating"].sum() * params["w"].pow(2).sum() + buffers["b"].sum()

    gradient_factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )
    result = metric_factory(
        vpx.Candidate(
            "metric",
            "row",
            {**metric_settings(), "dtype.output": "fp32"},
            admission_status="passed",
        ),
        {"metric_matrix": matrix},
        vector,
    )()

    assert tree_leaves(result)[0].dtype == torch.float32

    previous_precision = torch.get_float32_matmul_precision()
    previous_allow_bf16 = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    previous_allow_fp16 = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()

    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.use_deterministic_algorithms(False)
        gradient_factory(
            vpx.Candidate(
                "gradient",
                "row",
                {
                    **gradient_settings(),
                    "dtype.parameter_storage": "bf16",
                    "dtype.model_compute": "fp32",
                    "numeric.float32_matmul_precision": "high",
                    "numeric.bf16_reduced_precision_reduction": "true",
                    "numeric.fp16_reduced_precision_reduction": "true",
                    "numeric.deterministic_algorithms": "true",
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
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_allow_bf16
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            previous_allow_fp16
        )
        torch.use_deterministic_algorithms(previous_deterministic)

    assert observed == {
        "param_dtype": torch.float32,
        "buffer_dtype": torch.float32,
        "floating_dtype": torch.float32,
        "input_ids_dtype": torch.long,
        "attention_mask_dtype": torch.bool,
        "matmul_precision": "high",
        "allow_bf16": True,
        "allow_fp16": True,
        "deterministic": True,
    }
    assert torch.get_float32_matmul_precision() == previous_precision
    assert (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        is previous_allow_bf16
    )
    assert (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        is previous_allow_fp16
    )
    assert torch.are_deterministic_algorithms_enabled() is previous_deterministic

    with pytest.raises(vp.MaterializationError):
        gradient_factory(
            vpx.Candidate(
                "gradient",
                "row",
                {**gradient_settings(), "batch_size": 2},
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )()


def test_standard_runtime_executes_autodiff_compute_dtype_axis() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    buffers = {"b": torch.tensor([1.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert context.family == "gradient"
        observed["param_dtype"] = params["w"].dtype
        observed["buffer_dtype"] = buffers["b"].dtype
        observed["floating_dtype"] = batch["floating"].dtype

        return batch["floating"].sum() * params["w"].pow(2).sum() + buffers["b"].sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "autodiff-dtype",
            {**gradient_settings(), "dtype.autodiff_compute": "fp32"},
            admission_status="passed",
        ),
        {"floating": torch.tensor([2.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {
        "param_dtype": torch.float32,
        "buffer_dtype": torch.float32,
        "floating_dtype": torch.float32,
    }


def test_standard_runtime_applies_fp8_storage_before_model_compute() -> None:
    params = {"w": torch.tensor([1.3], dtype=torch.float32)}
    buffers = {"b": torch.tensor([0.7], dtype=torch.float32)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float32)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert context.family == "gradient"
        observed["param_dtype"] = params["w"].dtype
        observed["buffer_dtype"] = buffers["b"].dtype
        observed["batch_dtype"] = batch["scale"].dtype
        observed["param_value"] = params["w"].detach().clone()
        observed["buffer_value"] = buffers["b"].detach().clone()

        return (params["w"] + buffers["b"] + batch["scale"]).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers=buffers,
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "fp8-storage",
            {
                **gradient_settings(),
                "dtype.parameter_storage": "fp8_when_supported",
                "dtype.model_compute": "fp32",
                "dtype.output": "fp32",
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor([2.0], dtype=torch.float32)},
        vector,
    )()
    expected_param = (
        params["w"].to(dtype=runtime_module._fp8_dtype()).to(dtype=torch.float32)
    )
    expected_buffer = (
        buffers["b"].to(dtype=runtime_module._fp8_dtype()).to(dtype=torch.float32)
    )

    assert observed["param_dtype"] == torch.float32
    assert observed["buffer_dtype"] == torch.float32
    assert observed["batch_dtype"] == torch.float32
    torch.testing.assert_close(observed["param_value"], expected_param)
    torch.testing.assert_close(observed["buffer_value"], expected_buffer)
    assert tree_leaves(result)[0].dtype == torch.float32


def test_standard_runtime_executes_fp8_model_compute_boundary() -> None:
    params = {"w": torch.tensor([1.5], dtype=torch.float32)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float32)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        observed["param_dtype"] = params["w"].dtype
        observed["batch_dtype"] = batch["scale"].dtype

        return (params["w"].float() * batch["scale"].float()).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "fp8-compute",
            {
                **gradient_settings(),
                "dtype.model_compute": "fp8_when_supported",
                "dtype.output": "fp32",
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor([2.0], dtype=torch.float32)},
        vector,
    )()

    assert observed == {
        "param_dtype": runtime_module._fp8_dtype(),
        "batch_dtype": runtime_module._fp8_dtype(),
    }
    assert tree_leaves(result)[0].dtype == torch.float32


def test_standard_runtime_executes_split_model_and_autodiff_compute_dtypes() -> None:
    module = DtypeObservingStatefulModule()
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=dict(module.named_parameters()),
        buffers=dict(module.named_buffers()),
        module=module,
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "split-dtype-stateful",
            {
                **gradient_settings(),
                **stateful_module_call_settings(),
                "dtype.model_compute": "fp32",
                "dtype.autodiff_compute": "bf16",
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()

    assert module.observed == {
        "parameter": torch.float32,
        "buffer": torch.float32,
        "batch": torch.float32,
    }
    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([4.0], dtype=torch.bfloat16),
    )


def test_standard_runtime_rejects_split_dtypes_without_model_call() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([3.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="stateful_module"):
        factory(
            vpx.Candidate(
                "gradient",
                "split-dtype",
                {
                    **gradient_settings(),
                    "dtype.model_compute": "fp32",
                    "dtype.autodiff_compute": "bf16",
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_executes_non_reentrant_layer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []
    original_checkpoint_operation = runtime_module.checkpoint_operation

    def recording_checkpoint_operation(
        candidate: vpx.Candidate,
        function: Callable[..., vpx.TensorTree],
        args: Sequence[object],
        *,
        policy_key: str,
        activation_pack_hooks: Mapping[str, Callable[[torch.Tensor], object]]
        | None = None,
        activation_unpack_hooks: Mapping[str, Callable[[object], torch.Tensor]]
        | None = None,
        checkpoint_contexts: Mapping[str, Callable[[], object]] | None = None,
    ) -> vpx.CandidateOperation:
        assert activation_pack_hooks == {}
        assert activation_unpack_hooks == {}
        assert checkpoint_contexts == {}
        calls.append({
            "policy_key": policy_key,
            "recompute": candidate.settings["activation.recompute"],
            "context_fn": candidate.settings["checkpoint.context_fn"],
            "arg_count": len(args),
        })

        return original_checkpoint_operation(
            candidate,
            function,
            args,
            policy_key=policy_key,
            activation_pack_hooks=activation_pack_hooks,
            activation_unpack_hooks=activation_unpack_hooks,
            checkpoint_contexts=checkpoint_contexts,
        )

    monkeypatch.setattr(
        runtime_module,
        "checkpoint_operation",
        recording_checkpoint_operation,
    )
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "checkpoint-row",
            {**gradient_settings(), **standard_checkpoint_settings()},
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert calls == [
        {
            "policy_key": "activation.recompute",
            "recompute": "checkpoint_non_reentrant_by_layer",
            "context_fn": "none",
            "arg_count": 2,
        }
    ]
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_rejects_selective_checkpoint_without_context_pair() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    settings = {
        **standard_checkpoint_settings(),
        "activation.recompute": "checkpoint_selective",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="declared_context_pair"):
        factory(
            vpx.Candidate(
                "gradient",
                "checkpoint-selective",
                {**gradient_settings(), **settings},
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_executes_selective_checkpoint_with_context_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def context_fn() -> tuple[object, object]:
        return object(), object()

    def recording_checkpoint_operation(
        candidate: vpx.Candidate,
        function: Callable[..., vpx.TensorTree],
        args: Sequence[object],
        *,
        policy_key: str,
        activation_pack_hooks: Mapping[str, Callable[[torch.Tensor], object]]
        | None = None,
        activation_unpack_hooks: Mapping[str, Callable[[object], torch.Tensor]]
        | None = None,
        checkpoint_contexts: Mapping[str, Callable[[], object]] | None = None,
    ) -> vpx.CandidateOperation:
        assert activation_pack_hooks == {}
        assert activation_unpack_hooks == {}
        assert checkpoint_contexts == {"selective": context_fn}
        calls.append({
            "policy_key": policy_key,
            "recompute": candidate.settings["activation.recompute"],
            "context_fn": candidate.settings["checkpoint.context_fn"],
            "context_callable": candidate.settings["checkpoint.context_fn_callable"],
            "arg_count": len(args),
        })

        def operation() -> vpx.TensorTree:
            return function(*args)

        return operation

    monkeypatch.setattr(
        runtime_module,
        "checkpoint_operation",
        recording_checkpoint_operation,
    )
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        checkpoint_contexts={"selective": context_fn},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "checkpoint-selective",
            {
                **gradient_settings(),
                **standard_checkpoint_settings(),
                "activation.recompute": "checkpoint_selective",
                "checkpoint.context_fn": "declared_context_pair",
                "checkpoint.context_fn_callable": "selective",
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert calls == [
        {
            "policy_key": "activation.recompute",
            "recompute": "checkpoint_selective",
            "context_fn": "declared_context_pair",
            "context_callable": "selective",
            "arg_count": 2,
        }
    ]
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_rejects_manual_recompute_without_callback() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    settings = {
        **standard_checkpoint_settings(),
        "activation.recompute": "manual_recompute",
        "checkpoint.early_stop": "false",
        "checkpoint.preserve_rng_state": "false",
        "checkpoint.determinism_check": "none",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="declared recompute callback"):
        factory(
            vpx.Candidate(
                "gradient",
                "manual-recompute",
                {**gradient_settings(), **settings},
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_executes_manual_recompute_callback_override() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []

    def manual_recompute(
        candidate: vpx.Candidate,
        operation: vpx.CandidateOperation,
        args: tuple[torch.Tensor, ...],
    ) -> vpx.CandidateOperation:
        calls.append({
            "candidate_id": candidate.candidate_id,
            "arg_count": len(args),
        })

        def recomputed_operation() -> vpx.TensorTree:
            return operation()

        return recomputed_operation

    settings = {
        **standard_checkpoint_settings(),
        "activation.recompute": "manual_recompute",
        "checkpoint.early_stop": "false",
        "checkpoint.preserve_rng_state": "false",
        "checkpoint.determinism_check": "none",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        manual_recompute=manual_recompute,
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "manual-recompute",
            {**gradient_settings(), **settings},
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert calls == [{"candidate_id": "manual-recompute", "arg_count": 2}]
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_manual_recompute_executes_custom_saved_tensor_hooks() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    events = []

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append("pack")

        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append("unpack")

        return tensor

    def manual_recompute(
        candidate: vpx.Candidate,
        operation: vpx.CandidateOperation,
        args: tuple[torch.Tensor, ...],
    ) -> vpx.CandidateOperation:
        events.append(f"manual:{candidate.candidate_id}:{len(args)}")

        def recomputed_operation() -> vpx.TensorTree:
            return operation()

        return recomputed_operation

    settings = {
        **standard_checkpoint_settings(),
        "activation.recompute": "manual_recompute",
        "activation.offload": "custom_saved_tensor_hooks",
        "activation.pack_hook": "recording",
        "activation.unpack_hook": "recording",
        "checkpoint.early_stop": "false",
        "checkpoint.preserve_rng_state": "false",
        "checkpoint.determinism_check": "none",
    }
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        activation_pack_hooks={"recording": pack_hook},
        activation_unpack_hooks={"recording": unpack_hook},
        manual_recompute=manual_recompute,
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "manual-offload",
            {**gradient_settings(), **settings},
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert events == ["manual:manual-offload:2", "pack", "unpack"]
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_executes_custom_saved_tensor_hooks() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    events = []

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("pack", tensor.detach().clone()))

        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("unpack", tensor.detach().clone()))

        return tensor

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        activation_pack_hooks={"recording": pack_hook},
        activation_unpack_hooks={"recording": unpack_hook},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "activation-offload",
            {
                **gradient_settings(),
                "activation.recompute": "none",
                "activation.offload": "custom_saved_tensor_hooks",
                "activation.pack_hook": "recording",
                "activation.unpack_hook": "recording",
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert tuple(event for event, _ in events) == ("pack", "unpack")
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_executes_cpu_saved_tensor_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    events = []
    original_pack = runtime_module._cpu_pack_hook
    original_unpack = runtime_module._cpu_unpack_hook

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {"scale": 1.0}
        assert context.family == "gradient"

        return params["w"].pow(3).sum()

    def recording_pack(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
        events.append(("pack", tensor.device.type, tensor.detach().clone()))

        return original_pack(tensor)

    def recording_unpack(packed: tuple[torch.Tensor, torch.device]) -> torch.Tensor:
        tensor, device = packed
        events.append(("unpack", tensor.device.type, device.type))

        return original_unpack(packed)

    monkeypatch.setattr(runtime_module, "_cpu_pack_hook", recording_pack)
    monkeypatch.setattr(runtime_module, "_cpu_unpack_hook", recording_unpack)
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    result = factory(
        vpx.Candidate(
            "gradient",
            "cpu-saved-hooks",
            {
                **gradient_settings(),
                "activation.recompute": "none",
                "activation.offload": "saved_tensor_hooks_cpu",
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()
    result_map = tensor_mapping(result)

    assert tuple(event[0] for event in events) == ("pack", "unpack")
    assert events[0][1] == "cpu"
    assert events[1][1:] == ("cpu", "cpu")
    torch.testing.assert_close(
        result_map["w"],
        torch.tensor([12.0], dtype=torch.float64),
    )


def test_reference_check_rejects_custom_saved_tensor_hooks_that_change_values() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor)

    check = vpx.standard_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "directional_abs_diff": 1e-12,
            "directional_rel_diff": 1e-12,
        },
        scalar_objectives={"loss": quadratic_scalar},
        activation_pack_hooks={"zero": pack_hook},
        activation_unpack_hooks={"zero": unpack_hook},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "gradient",
                "bad-hooks",
                {
                    **gradient_settings(),
                    "activation.recompute": "none",
                    "activation.offload": "custom_saved_tensor_hooks",
                    "activation.pack_hook": "zero",
                    "activation.unpack_hook": "zero",
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_activation_offload_preserves_higher_order_hvp() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    events = []

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append("pack")

        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append("unpack")

        return tensor

    factory = vpx.standard_operation_factory(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        activation_pack_hooks={"recording": pack_hook},
        activation_unpack_hooks={"recording": unpack_hook},
    )
    result = factory(
        vpx.Candidate(
            "hvp",
            "activation-offload",
            {
                **hvp_settings("reverse_over_reverse"),
                "activation.recompute": "none",
                "activation.offload": "custom_saved_tensor_hooks",
                "activation.pack_hook": "recording",
                "activation.unpack_hook": "recording",
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        vector,
    )()

    assert "pack" in events
    assert "unpack" in events
    torch.testing.assert_close(
        tree_leaves(result)[0],
        torch.tensor([2.0], dtype=torch.float64),
    )


def test_higher_order_reference_rejects_custom_hooks_that_change_values() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64, requires_grad=True)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor)

    check = vpx.standard_reference_check(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params=params,
        buffers={},
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "symmetry_max_abs_diff": 1e-12,
            "directional_abs_diff": 1e-12,
            "directional_rel_diff": 1e-12,
        },
        scalar_objectives={"loss": quadratic_scalar},
        activation_pack_hooks={"zero": pack_hook},
        activation_unpack_hooks={"zero": unpack_hook},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "hvp",
                "bad-hooks",
                {
                    **hvp_settings("reverse_over_reverse"),
                    "activation.recompute": "none",
                    "activation.offload": "custom_saved_tensor_hooks",
                    "activation.pack_hook": "zero",
                    "activation.unpack_hook": "zero",
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


@pytest.mark.parametrize(
    ("host_to_device", "calls_before", "calls_after"),
    [
        (
            "outside_measured_call",
            ["outside_measured_call"],
            ["outside_measured_call"],
        ),
        (
            "inside_measured_call",
            [],
            ["inside_measured_call"],
        ),
    ],
)
def test_standard_runtime_moves_inputs_at_declared_call_boundary(
    monkeypatch: pytest.MonkeyPatch,
    host_to_device: str,
    calls_before: list[str],
    calls_after: list[str],
) -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    calls = []
    original = runtime_module._runtime_batch_input_residency

    def recording_input_residency(
        batch: vpx.Batch,
        settings: Mapping[str, Any],
    ) -> vpx.Batch:
        calls.append(settings["input.host_to_device"])

        return original(batch, settings)

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        scale = batch["scale_tensor"]
        assert isinstance(scale, torch.Tensor)

        return params["w"].pow(2).sum() * scale.sum()

    monkeypatch.setattr(
        runtime_module,
        "_runtime_batch_input_residency",
        recording_input_residency,
    )
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            f"input-{host_to_device}",
            {
                **gradient_settings(),
                "input.residency": "cpu_staged",
                "input.host_to_device": host_to_device,
            },
            admission_status="passed",
        ),
        {"scale_tensor": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )

    assert calls == calls_before

    result = operation()
    result_map = tensor_mapping(result)

    assert calls == calls_after
    assert torch.equal(result_map["w"], torch.tensor([4.0], dtype=torch.float64))


def test_standard_runtime_moves_inputs_to_pinned_cpu() -> None:
    try:
        torch.empty(1).pin_memory()
    except RuntimeError as error:
        pytest.skip(str(error))

    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        scale = batch["scale_tensor"]
        assert isinstance(scale, torch.Tensor)
        observed["is_pinned"] = scale.is_pinned()

        return params["w"].pow(2).sum() * scale.sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "input-pinned",
            {
                **gradient_settings(),
                "input.residency": "cpu_pinned",
                "input.host_to_device": "outside_measured_call",
            },
            admission_status="passed",
        ),
        {"scale_tensor": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {"is_pinned": True}


def test_standard_runtime_moves_inputs_to_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for input.residency=gpu")

    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        scale = batch["scale_tensor"]
        assert isinstance(scale, torch.Tensor)
        observed["device"] = scale.device.type

        return params["w"].pow(2).sum() * scale.cpu().sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "input-gpu",
            {
                **gradient_settings(),
                "input.residency": "gpu",
                "input.host_to_device": "outside_measured_call",
            },
            admission_status="passed",
        ),
        {"scale_tensor": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {"device": "cuda"}


def test_standard_runtime_executes_precomputed_cpu_teacher_outputs() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        teacher = batch["teacher_outputs"]
        assert isinstance(teacher, torch.Tensor)
        observed["device"] = teacher.device.type

        return params["w"].pow(2).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "teacher-cpu",
            {**gradient_settings(), "teacher_outputs": "precomputed_cpu"},
            admission_status="passed",
        ),
        {"teacher_outputs": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {"device": "cpu"}


def test_standard_runtime_executes_precomputed_pinned_teacher_outputs() -> None:
    try:
        torch.empty(1).pin_memory()
    except RuntimeError as error:
        pytest.skip(str(error))

    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        teacher = batch["teacher_outputs"]
        assert isinstance(teacher, torch.Tensor)
        observed["is_pinned"] = teacher.is_pinned()

        return params["w"].pow(2).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "teacher-pinned",
            {**gradient_settings(), "teacher_outputs": "precomputed_cpu_pinned"},
            admission_status="passed",
        ),
        {"teacher_outputs": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {"is_pinned": True}


def test_standard_runtime_executes_precomputed_gpu_teacher_outputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for precomputed GPU teacher outputs")

    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    observed = {}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        teacher = batch["teacher_outputs"]
        assert isinstance(teacher, torch.Tensor)
        observed["device"] = teacher.device.type

        return params["w"].pow(2).sum()

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
    )
    factory(
        vpx.Candidate(
            "gradient",
            "teacher-gpu",
            {**gradient_settings(), "teacher_outputs": "precomputed_gpu"},
            admission_status="passed",
        ),
        {"teacher_outputs": torch.tensor([1.0], dtype=torch.float64)},
        vector,
    )()

    assert observed == {"device": "cuda"}


def test_standard_runtime_teacher_outputs_require_fixed_batch_field() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="teacher_outputs"):
        factory(
            vpx.Candidate(
                "gradient",
                "teacher-missing",
                {**gradient_settings(), "teacher_outputs": "precomputed_cpu"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            vector,
        )


def test_standard_runtime_recomputed_teacher_outputs_need_objective() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )

    with pytest.raises(vp.MaterializationError, match="teacher objective"):
        factory(
            vpx.Candidate(
                "gradient",
                "teacher-recompute",
                {
                    **gradient_settings(),
                    "teacher_outputs": "recomputed_with_equality_check",
                },
                admission_status="passed",
            ),
            {"teacher_outputs": torch.tensor([1.0], dtype=torch.float64)},
            vector,
        )


def test_standard_runtime_recomputes_teacher_outputs_inside_operation() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    events = []

    def teacher_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert params["w"] is not None
        assert buffers == {}
        assert context.family == "gradient"
        events.append("teacher")

        return {"logits": batch["teacher_seed"] + 1.0}

    def scalar(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"
        teacher = batch["teacher_outputs"]
        assert isinstance(teacher, Mapping)
        logits = teacher["logits"]
        assert isinstance(logits, torch.Tensor)
        events.append("scalar")

        return params["w"].pow(2).sum() + logits.sum() * 0.0

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": scalar},
        teacher_objective=teacher_objective,
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            "teacher-recompute",
            {
                **gradient_settings(),
                "teacher_outputs": "recomputed_with_equality_check",
            },
            admission_status="passed",
        ),
        {
            "teacher_seed": torch.tensor([2.0], dtype=torch.float64),
            "teacher_outputs": {"logits": torch.tensor([3.0], dtype=torch.float64)},
        },
        vector,
    )

    assert events == []
    result = operation()
    result_map = tensor_mapping(result)

    assert events == ["teacher", "scalar"]
    torch.testing.assert_close(
        result_map["w"],
        torch.tensor([4.0], dtype=torch.float64),
    )


def test_standard_runtime_rejects_mismatched_recomputed_teacher_outputs() -> None:
    params = {"w": torch.tensor([2.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    def teacher_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> vpx.TensorTree:
        assert params["w"] is not None
        assert buffers == {}
        assert batch["teacher_outputs"]
        assert context.family == "gradient"

        return {"logits": torch.tensor([4.0], dtype=torch.float64)}

    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params=params,
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
        teacher_objective=teacher_objective,
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            "teacher-recompute",
            {
                **gradient_settings(),
                "teacher_outputs": "recomputed_with_equality_check",
            },
            admission_status="passed",
        ),
        {"teacher_outputs": {"logits": torch.tensor([3.0], dtype=torch.float64)}},
        vector,
    )

    with pytest.raises(vp.MaterializationError, match="do not match"):
        operation()


def test_standard_runtime_rejects_cuda_autocast_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")

    factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="CUDA autocast requires CUDA"):
        factory(
            vpx.Candidate(
                "metric",
                "autocast",
                {**metric_settings(), "autocast": "cuda_bf16"},
                admission_status="passed",
            ),
            {"metric_matrix": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )()


@pytest.mark.parametrize(
    ("autocast", "expected_dtype"),
    [("cuda_fp16", torch.float16), ("cuda_bf16", torch.bfloat16)],
)
def test_standard_runtime_enters_cuda_autocast_context(
    monkeypatch: pytest.MonkeyPatch,
    autocast: str,
    expected_dtype: torch.dtype,
) -> None:
    events = []

    class FakeAutocast:
        def __init__(self, device_type: str, dtype: torch.dtype) -> None:
            self.device_type = device_type
            self.dtype = dtype

        def __enter__(self) -> None:
            events.append(("enter", self.device_type, self.dtype))

        def __exit__(self, *args: object) -> None:
            events.append(("exit", self.device_type, self.dtype))

    def fake_autocast(*, device_type: str, dtype: torch.dtype) -> FakeAutocast:
        return FakeAutocast(device_type, dtype)

    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_module.torch, "autocast", fake_autocast)
    factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
    )
    result = factory(
        vpx.Candidate(
            "metric",
            "autocast",
            {**metric_settings(), "autocast": autocast},
            admission_status="passed",
        ),
        {"metric_matrix": torch.eye(1, dtype=torch.float64)},
        {"w": torch.tensor([2.0], dtype=torch.float64)},
    )()

    assert torch.equal(tree_leaves(result)[0], torch.tensor([2.0], dtype=torch.float64))
    assert events == [
        ("enter", "cuda", expected_dtype),
        ("exit", "cuda", expected_dtype),
    ]


def test_standard_runtime_compiles_whole_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = {"w": torch.tensor([3.0, 4.0], dtype=torch.float64)}
    events = []
    recorder = CompileRecorder(
        events,
        called_event="compiled_call",
        expected_mode=None,
        expected_options={"triton.cudagraphs": True},
    )
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operation = compiled_metric_operation(
        {
            **metric_settings(),
            **compile_settings(mode=None, cuda_graphs="true"),
        },
        params={"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        batch={"metric_matrix": torch.eye(2, dtype=torch.float64)},
        vector=vector,
    )

    assert events == [("compile", False)]
    torch.testing.assert_close(tree_leaves(operation())[0], vector["w"])
    assert events[-1] == ("compiled_call", True)


def test_standard_runtime_compiles_stateful_model_forward_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    def model_forward_event(batch: vpx.Batch) -> tuple[str, tuple[str, ...]]:
        return "compiled_model_forward", tuple(batch)

    recorder = CompileRecorder(events, call_event=model_forward_event)
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operation = compiled_stateful_model_forward_operation(
        compile_settings(boundary="model_forward"),
    )

    assert events == [("compile", False)]
    torch.testing.assert_close(
        tensor_mapping(operation())["w"],
        torch.tensor([4.0], dtype=torch.float64),
    )
    assert events[-1] == ("compiled_model_forward", ("scale",))


@pytest.mark.parametrize("fullgraph", ["false", "true"])
def test_standard_runtime_runs_real_torch_compile_whole_operator(
    fullgraph: str,
) -> None:
    vector = {"w": torch.tensor([3.0, 4.0], dtype=torch.float64)}
    assert_compiled_boundary_result(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params={"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        settings={
            **metric_settings(),
            **compile_settings(),
            "compile.fullgraph": fullgraph,
        },
        batch={"metric_matrix": torch.eye(2, dtype=torch.float64)},
        vector=vector,
        expected=vector["w"],
    )


@pytest.mark.parametrize("boundary", ["gradient_closure", "loss_closure"])
def test_standard_runtime_runs_real_torch_compile_gradient_boundary(
    boundary: str,
) -> None:
    assert_compiled_boundary_result(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        settings={
            **gradient_settings(),
            **compile_settings(boundary=boundary),
        },
        batch={"scale": 3.0},
        vector={"w": torch.tensor([1.0], dtype=torch.float64)},
        scalar_objectives={"loss": quadratic_scalar},
        expected=torch.tensor([12.0], dtype=torch.float64),
    )


def test_standard_runtime_runs_real_torch_compile_model_forward() -> None:
    operation = compiled_stateful_model_forward_operation(
        compile_settings(boundary="model_forward"),
    )
    result = operation()

    torch.testing.assert_close(
        tensor_mapping(result)["w"],
        torch.tensor([4.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("operator", "settings", "batch", "vector", "expected"),
    [
        (
            ops.metric(
                "metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
            ),
            {
                **metric_settings(),
                **compile_settings(boundary="metric_multiply"),
            },
            {
                "metric_matrix": torch.tensor(
                    [[2.0, 0.5], [0.5, 3.0]],
                    dtype=torch.float64,
                )
            },
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            torch.tensor([3.0, 6.5], dtype=torch.float64),
        ),
        (
            ops.inverse_metric(
                "inverse",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
                damping=0.0,
            ),
            {
                **inverse_metric_settings(),
                **compile_settings(boundary="inverse_metric_solve"),
            },
            {
                "metric_matrix": torch.tensor(
                    [[2.0, 0.0], [0.0, 4.0]],
                    dtype=torch.float64,
                )
            },
            {"w": torch.tensor([6.0, 8.0], dtype=torch.float64)},
            torch.tensor([3.0, 2.0], dtype=torch.float64),
        ),
    ],
)
def test_standard_runtime_runs_real_torch_compile_linear_algebra_boundaries(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: dict[str, torch.Tensor],
    expected: torch.Tensor,
) -> None:
    assert_compiled_boundary_result(
        operator,
        params={"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        settings=settings,
        batch=batch,
        vector=vector,
        expected=expected,
    )


@pytest.mark.parametrize(
    ("operator", "settings"),
    [
        (
            ops.jvp("jvp", "function", aggregation="sum"),
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                **compile_settings(boundary="jvp_closure"),
            },
        ),
        (
            ops.vjp("vjp", "function", aggregation="sum"),
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                **compile_settings(boundary="vjp_closure"),
            },
        ),
    ],
)
def test_standard_runtime_runs_real_torch_compile_linear_function_boundary(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
) -> None:
    assert_compiled_boundary_result(
        operator,
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        settings=settings,
        batch={"scale": torch.tensor([2.0], dtype=torch.float64)},
        vector={"w": torch.tensor([3.0], dtype=torch.float64)},
        function_objectives={"function": scaled_tree_function},
        expected=torch.tensor([6.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("settings", "vector", "expected"),
    [
        (
            {
                **hvp_settings("jvp_grad"),
                **torch_func_settings(requires_forward_ad=True),
                **compile_settings(boundary="hvp_single_vector"),
            },
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            torch.tensor([6.0, 12.0], dtype=torch.float64),
        ),
        (
            {
                **hvp_settings("jvp_grad"),
                **torch_func_settings(requires_forward_ad=True),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
                **compile_settings(boundary="hvp_batched_vectors"),
            },
            {
                "w": torch.tensor(
                    [[1.0, 0.0], [0.0, 2.0]],
                    dtype=torch.float64,
                )
            },
            torch.tensor([[6.0, 0.0], [0.0, 12.0]], dtype=torch.float64),
        ),
    ],
)
def test_standard_runtime_runs_real_torch_compile_hvp_boundary(
    settings: Mapping[str, object],
    vector: vpx.TensorTree,
    expected: torch.Tensor,
) -> None:
    assert_compiled_boundary_result(
        ops.hvp("hvp", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
        settings=settings,
        batch={"scale": 3.0},
        vector=vector,
        scalar_objectives={"loss": quadratic_scalar},
        expected=expected,
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "ggn_full_product",
        "ggn_loss_hessian_product",
        "ggn_jvp",
        "ggn_vjp",
    ],
)
def test_standard_runtime_runs_real_torch_compile_ggn_boundary(
    boundary: str,
) -> None:
    assert_compiled_boundary_result(
        ops.ggnvp("ggn", "model_output", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        settings={
            **ggn_dense_kernel_settings(),
            **compile_settings(boundary=boundary),
        },
        batch={
            "scale": 1.0,
            "loss_hessian": torch.tensor([[5.0]], dtype=torch.float64),
        },
        vector={"w": torch.tensor([3.0], dtype=torch.float64)},
        function_objectives={"model_output": square_function},
        expected=torch.tensor([240.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("operator", "settings", "batch", "expected"),
    [
        (
            score_terms_fisher("fisher", "scores"),
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                "schedule.per_example": "loop",
                **compile_settings(boundary="fisher_score_grad"),
            },
            {
                "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                "normalization": 3.0,
            },
            torch.tensor([14.0 / 3.0, 28.0 / 3.0], dtype=torch.float64),
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            {
                **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
                **compile_settings(boundary="sampled_fisher_score_grad"),
            },
            {
                "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                "normalization": 3.0,
                "num_examples": 3.0,
            },
            torch.tensor([7.0 / 3.0, 14.0 / 3.0], dtype=torch.float64),
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="num_examples",
            ),
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
                **compile_settings(boundary="empirical_fisher_example_grad"),
            },
            {
                "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                "normalization": 3.0,
                "num_examples": 3.0,
            },
            torch.tensor([14.0 / 3.0, 28.0 / 3.0], dtype=torch.float64),
        ),
        (
            ops.per_example_gradient(
                "per_example",
                "scores",
                aggregation="sum",
                example_loss_reduction="per_example",
            ),
            {
                **per_example_gradient_settings(),
                **compile_settings(boundary="per_example_gradient"),
            },
            {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
            torch.tensor(
                [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_standard_runtime_runs_real_torch_compile_score_grad_boundary(
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    expected: torch.Tensor,
) -> None:
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([1.0, -1.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"scores": linear_score_rows},
    )
    operation = factory(
        vpx.Candidate(
            operator.family,
            "compiled-score",
            settings,
            admission_status="passed",
        ),
        batch,
        {"w": torch.tensor([0.5, 0.25], dtype=torch.float64)},
    )
    result = operation()

    torch.testing.assert_close(tree_leaves(result)[0], expected)


def test_standard_runtime_warms_compile_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = {"w": torch.tensor([3.0, 4.0], dtype=torch.float64)}
    events = []
    recorder = CompileRecorder(events, called_event="called")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operation = compiled_metric_operation(
        {
            **metric_settings(),
            **compile_settings(cache_state="warm_cache"),
        },
        params={"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        batch={"metric_matrix": torch.eye(2, dtype=torch.float64)},
        vector=vector,
    )

    assert events == [("compile", False), ("called", True)]
    assert torch.allclose(tree_leaves(operation())[0], vector["w"])
    assert events == [
        ("compile", False),
        ("called", True),
        ("called", True),
    ]


def test_standard_runtime_accepts_operator_specific_compile_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event="called")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operation = compiled_metric_operation(
        {
            **metric_settings(),
            **compile_settings(boundary="metric_multiply"),
        },
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        batch={"metric_matrix": torch.eye(1, dtype=torch.float64)},
        vector={"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert torch.allclose(
        tree_leaves(operation())[0],
        torch.tensor([1.0], dtype=torch.float64),
    )
    assert events == [("compile", False), ("called", True)]


@pytest.mark.parametrize(
    (
        "operator",
        "settings",
        "batch",
        "vector",
        "expected",
        "finite_name",
    ),
    [
        (
            ops.metric(
                "metric",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
            ),
            {
                **metric_settings(),
                **compile_settings(boundary="metric_multiply"),
            },
            {"metric_matrix": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
            torch.tensor([1.0], dtype=torch.float64),
            "metric result",
        ),
        (
            ops.inverse_metric(
                "inverse",
                "dense",
                aggregation="sum",
                representation=dense_metric_representation(),
                damping=0.0,
            ),
            {
                **inverse_metric_settings(),
                **compile_settings(boundary="inverse_metric_solve"),
            },
            {"metric_matrix": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
            torch.tensor([1.0], dtype=torch.float64),
            "inverse metric result",
        ),
    ],
)
def test_standard_runtime_compiles_metric_tree_boundary_only(
    monkeypatch: pytest.MonkeyPatch,
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    expected: torch.Tensor,
    finite_name: str,
) -> None:
    events = []
    recorder = CompileRecorder(events)
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    patch_require_finite_tree_recorder(monkeypatch, recorder)
    operation = compiled_boundary_operation(
        operator,
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        settings=settings,
        batch=batch,
        vector=vector,
    )

    assert events == [("compile", False)]
    torch.testing.assert_close(tree_leaves(operation())[0], expected)
    assert events == [
        ("compile", False),
        ("finite_tree", finite_name, False),
    ]


@pytest.mark.parametrize(
    (
        "operator",
        "params",
        "settings",
        "batch",
        "vector",
        "expected",
        "scalar_objectives",
        "function_objectives",
        "tag",
        "expected_events",
    ),
    [
        (
            ops.gradient("gradient", "loss", aggregation="sum"),
            {"w": torch.tensor([2.0], dtype=torch.float64)},
            {
                **gradient_settings(),
                **compile_settings(boundary="gradient_closure"),
            },
            {"scale": 3.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
            torch.tensor([12.0], dtype=torch.float64),
            {"loss": quadratic_scalar},
            None,
            None,
            (("compile", False), ("runtime_output", False)),
        ),
        (
            ops.jvp("jvp", "function", aggregation="sum"),
            {"w": torch.tensor([2.0], dtype=torch.float64)},
            {
                **jvp_settings("torch_func_jvp"),
                **torch_func_settings(requires_forward_ad=True),
                **compile_settings(boundary="jvp_closure"),
            },
            {"scale": torch.tensor([2.0], dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            torch.tensor([6.0], dtype=torch.float64),
            None,
            {"function": scaled_tree_function},
            None,
            (("compile", False), ("runtime_output", False)),
        ),
        (
            ops.vjp("vjp", "function", aggregation="sum"),
            {"w": torch.tensor([2.0], dtype=torch.float64)},
            {
                **vjp_settings(),
                **torch_func_settings(requires_forward_ad=False),
                **compile_settings(boundary="vjp_closure"),
            },
            {"scale": torch.tensor([2.0], dtype=torch.float64)},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            torch.tensor([6.0], dtype=torch.float64),
            None,
            {"function": scaled_tree_function},
            None,
            (("compile", False), ("runtime_output", False)),
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
            {
                **hvp_settings("reverse_over_reverse"),
                **compile_settings(boundary="hvp_single_vector"),
            },
            {"scale": 3.0},
            {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)},
            torch.tensor([6.0, 0.0], dtype=torch.float64),
            {"loss": quadratic_scalar},
            None,
            "hvp_single_vector",
            (
                ("compile", "hvp_single_vector", False),
                ("runtime_output", "hvp_single_vector", False),
            ),
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
                **compile_settings(boundary="hvp_batched_vectors"),
            },
            {"scale": 3.0},
            {
                "w": torch.tensor(
                    [[1.0, 0.0], [0.0, 2.0]],
                    dtype=torch.float64,
                )
            },
            torch.tensor([[6.0, 0.0], [0.0, 12.0]], dtype=torch.float64),
            {"loss": quadratic_scalar},
            None,
            "hvp_batched_vectors",
            (
                ("compile", "hvp_batched_vectors", False),
                ("runtime_output", "hvp_batched_vectors", False),
            ),
        ),
        (
            ops.ggnvp("ggn", "model_output", aggregation="sum"),
            {"w": torch.tensor([2.0], dtype=torch.float64)},
            {
                **ggn_dense_kernel_settings(),
                **compile_settings(boundary="ggn_full_product"),
            },
            {
                "scale": 1.0,
                "loss_hessian": torch.tensor([[5.0]], dtype=torch.float64),
            },
            {"w": torch.tensor([3.0], dtype=torch.float64)},
            torch.tensor([240.0], dtype=torch.float64),
            None,
            {"model_output": square_function},
            None,
            (("compile", False), ("runtime_output", False)),
        ),
    ],
)
def test_standard_runtime_compiles_runtime_output_boundary_only(
    monkeypatch: pytest.MonkeyPatch,
    operator: vpx.OperatorSpec,
    params: vpx.ParameterTree,
    settings: Mapping[str, object],
    batch: vpx.Batch,
    vector: vpx.TensorTree,
    expected: torch.Tensor,
    scalar_objectives: Mapping[str, Callable[..., torch.Tensor]] | None,
    function_objectives: Mapping[str, Callable[..., vpx.TensorTree]] | None,
    tag: str | None,
    expected_events: tuple[tuple[Any, ...], ...],
) -> None:
    events = []
    recorder = CompileRecorder(events, tag=tag)
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    patch_runtime_output_recorder(monkeypatch, recorder)
    operation = compiled_boundary_operation(
        operator,
        params=params,
        settings=settings,
        batch=batch,
        vector=vector,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )

    assert events == [expected_events[0]]
    torch.testing.assert_close(tree_leaves(operation())[0], expected)
    assert tuple(events) == expected_events


@pytest.mark.parametrize(
    (
        "operator",
        "params",
        "settings",
        "vector",
        "expected",
        "expected_events",
    ),
    [
        (
            ops.gradient("gradient", "loss", aggregation="sum"),
            {"w": torch.tensor([2.0], dtype=torch.float64)},
            {
                **gradient_settings(),
                **compile_settings(boundary="loss_closure"),
            },
            {"w": torch.tensor([1.0], dtype=torch.float64)},
            torch.tensor([12.0], dtype=torch.float64),
            (
                ("compile", False),
                ("compiled_loss", True),
                ("grad", False),
                ("runtime_output", False),
            ),
        ),
        (
            ops.hvp("hvp", "loss", aggregation="sum"),
            {"w": torch.tensor([2.0, -1.0], dtype=torch.float64)},
            {
                **hvp_settings("reverse_over_reverse"),
                **compile_settings(boundary="loss_closure"),
            },
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            torch.tensor([6.0, 12.0], dtype=torch.float64),
            (
                ("compile", False),
                ("compiled_loss", True),
                ("grad", False),
                ("grad", False),
                ("runtime_output", False),
            ),
        ),
    ],
)
def test_standard_runtime_compiles_loss_closure_boundary_only(
    monkeypatch: pytest.MonkeyPatch,
    operator: vpx.OperatorSpec,
    params: vpx.ParameterTree,
    settings: Mapping[str, object],
    vector: vpx.TensorTree,
    expected: torch.Tensor,
    expected_events: tuple[tuple[Any, ...], ...],
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event="compiled_loss")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    patch_autograd_grad_recorder(monkeypatch, recorder)
    patch_runtime_output_recorder(monkeypatch, recorder)
    operation = compiled_boundary_operation(
        operator,
        params=params,
        settings=settings,
        batch={"scale": 3.0},
        vector=vector,
        scalar_objectives={"loss": quadratic_scalar},
    )

    assert events == [("compile", False)]
    torch.testing.assert_close(tree_leaves(operation())[0], expected)
    assert tuple(events) == expected_events


@pytest.mark.parametrize(
    (
        "settings",
        "compiled_event",
        "patch_loss_product",
        "patch_vjp",
        "expected_events",
    ),
    [
        (
            {
                **ggn_dense_kernel_settings(),
                **compile_settings(boundary="ggn_jvp"),
            },
            "compiled_jvp",
            True,
            True,
            (
                ("compile", False),
                ("compiled_jvp", True),
                ("loss_hessian", False),
                ("vjp", False),
                ("runtime_output", False),
            ),
        ),
        (
            {
                **ggn_dense_kernel_settings(),
                **compile_settings(boundary="ggn_loss_hessian_product"),
            },
            "compiled_loss_product",
            False,
            True,
            (
                ("compile", False),
                ("compiled_loss_product", True),
                ("vjp", False),
                ("runtime_output", False),
            ),
        ),
        (
            {
                **ggn_dense_kernel_settings(),
                **compile_settings(boundary="ggn_vjp"),
            },
            "compiled_vjp",
            True,
            False,
            (
                ("compile", False),
                ("loss_hessian", False),
                ("compiled_vjp", True),
                ("runtime_output", False),
            ),
        ),
    ],
)
def test_standard_runtime_compiles_ggn_internal_boundary_only(
    monkeypatch: pytest.MonkeyPatch,
    settings: Mapping[str, object],
    compiled_event: str,
    patch_loss_product: bool,
    patch_vjp: bool,
    expected_events: tuple[tuple[Any, ...], ...],
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event=compiled_event)
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)

    if patch_loss_product:
        patch_ggn_loss_product_recorder(monkeypatch, recorder)

    if patch_vjp:
        patch_ggn_vjp_recorder(monkeypatch, recorder)

    patch_runtime_output_recorder(monkeypatch, recorder)
    operation = compiled_ggn_boundary_operation(settings)

    assert events == [("compile", False)]
    torch.testing.assert_close(
        tree_leaves(operation())[0],
        torch.tensor([240.0], dtype=torch.float64),
    )
    assert tuple(events) == expected_events


def test_standard_runtime_warms_ggn_loss_product_compile_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event="compiled_loss_product")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operation = compiled_ggn_boundary_operation(
        {
            **ggn_dense_kernel_settings(),
            **compile_settings(
                boundary="ggn_loss_hessian_product",
                cache_state="warm_cache",
            ),
        },
    )

    assert events == [
        ("compile", False),
        ("compiled_loss_product", True),
    ]
    torch.testing.assert_close(
        tree_leaves(operation())[0],
        torch.tensor([240.0], dtype=torch.float64),
    )
    assert events == [
        ("compile", False),
        ("compiled_loss_product", True),
        ("compiled_loss_product", True),
    ]


@pytest.mark.parametrize(
    (
        "operator",
        "settings",
        "boundary",
        "expected",
    ),
    [
        (
            score_terms_fisher("fisher", "scores"),
            {
                **fisher_settings(
                    "streaming_dot_accumulate",
                    score_grad_path="torch_autograd_grad_loop",
                ),
                "schedule.per_example": "loop",
                **compile_settings(boundary="fisher_score_grad"),
            },
            "fisher_score_grad",
            torch.tensor([14.0 / 3.0, 28.0 / 3.0], dtype=torch.float64),
        ),
        (
            score_terms_sampled_fisher("sampled", "scores"),
            {
                **sampled_fisher_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
                **compile_settings(boundary="sampled_fisher_score_grad"),
            },
            "sampled_fisher_score_grad",
            torch.tensor([7.0 / 3.0, 14.0 / 3.0], dtype=torch.float64),
        ),
        (
            ops.empirical_fisher_vp(
                "empirical",
                "scores",
                aggregation="mean_per_example",
                example_loss_reduction="per_example",
                denominator="num_examples",
            ),
            {
                **empirical_grad_settings("torch_autograd_grad_loop"),
                "schedule.per_example": "loop",
                **compile_settings(boundary="empirical_fisher_example_grad"),
            },
            "empirical_fisher_example_grad",
            torch.tensor([14.0 / 3.0, 28.0 / 3.0], dtype=torch.float64),
        ),
    ],
)
def test_standard_runtime_compiles_score_matrix_boundary_only(
    monkeypatch: pytest.MonkeyPatch,
    operator: vpx.OperatorSpec,
    settings: Mapping[str, object],
    boundary: str,
    expected: torch.Tensor,
) -> None:
    events = []
    recorder = CompileRecorder(
        events,
        tag=boundary,
        called_event="compiled_score_matrix",
    )

    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    patch_score_matrix_product_recorder(monkeypatch, recorder)
    patch_runtime_output_recorder(monkeypatch, recorder)
    factory = vpx.standard_operation_factory(
        operator,
        params={"w": torch.tensor([1.0, -1.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"scores": linear_score_rows},
    )
    operation = factory(
        vpx.Candidate(
            operator.family,
            "compiled",
            settings,
            admission_status="passed",
        ),
        {
            "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "normalization": 3.0,
            "num_examples": 3.0,
        },
        {"w": torch.tensor([0.5, 0.25], dtype=torch.float64)},
    )

    assert events == [("compile", boundary, False)]
    torch.testing.assert_close(tree_leaves(operation())[0], expected)
    assert events == [
        ("compile", boundary, False),
        ("compiled_score_matrix", boundary, True),
        ("runtime_output", boundary, False),
    ]


def test_standard_runtime_rejects_compile_boundary_without_matching_callable() -> None:
    factory = quadratic_hvp_factory({"w": torch.tensor([1.0], dtype=torch.float64)})

    with pytest.raises(vp.MaterializationError, match="hvp_batched_vectors"):
        factory(
            passed_candidate(
                "hvp",
                "compiled",
                {
                    **hvp_settings("jvp_grad"),
                    **torch_func_settings(requires_forward_ad=True),
                    **compile_settings(boundary="hvp_batched_vectors"),
                },
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_rejects_loss_closure_boundary_without_scalar_loss() -> None:
    factory = vpx.standard_operation_factory(
        ops.jvp("jvp", "function", aggregation="sum"),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
        function_objectives={"function": square_function},
    )

    with pytest.raises(vp.MaterializationError, match="loss_closure"):
        factory(
            vpx.Candidate(
                "jvp",
                "bad-loss-boundary",
                {
                    **jvp_settings("torch_func_jvp"),
                    **torch_func_settings(requires_forward_ad=True),
                    **compile_settings(boundary="loss_closure"),
                },
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_standard_runtime_compiles_hvp_batched_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event="called")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    factory = quadratic_hvp_factory({
        "w": torch.tensor([2.0, -1.0], dtype=torch.float64)
    })
    vector = {
        "w": torch.tensor(
            [[1.0, 0.0], [0.0, 2.0]],
            dtype=torch.float64,
        )
    }
    operation = factory(
        passed_candidate(
            "hvp",
            "compiled-batched",
            {
                **hvp_settings("reverse_over_reverse"),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
                **compile_settings(boundary="hvp_batched_vectors"),
            },
        ),
        {"scale": 3.0},
        vector,
    )
    result = operation()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"] * 6.0)
    assert events == [("compile", False), ("called", True)]


def test_standard_runtime_rejects_single_vector_compile_boundary_for_batched_hvp() -> (
    None
):
    factory = quadratic_hvp_factory({"w": torch.tensor([1.0], dtype=torch.float64)})

    with pytest.raises(vp.MaterializationError, match="hvp_single_vector"):
        factory(
            passed_candidate(
                "hvp",
                "compiled-bad-boundary",
                {
                    **hvp_settings("reverse_over_reverse"),
                    "vectorization.mode": "single_loop",
                    "vectorization.in_dims": {"w": 0},
                    **compile_settings(boundary="hvp_single_vector"),
                },
            ),
            {"scale": 1.0},
            {"w": torch.tensor([[1.0]], dtype=torch.float64)},
        )


def test_standard_runtime_enables_compiled_autograd_for_backward_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    original_compiled_autograd = torch_dynamo_config.compiled_autograd

    def compile_event() -> tuple[str, bool]:
        return "compile", torch_dynamo_config.compiled_autograd

    def call_event() -> tuple[str, bool]:
        return "call", torch_dynamo_config.compiled_autograd

    recorder = CompileRecorder(
        events,
        compile_event=compile_event,
        call_event=call_event,
    )
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    factory = vpx.standard_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([2.0], dtype=torch.float64)},
        buffers={},
        scalar_objectives={"loss": quadratic_scalar},
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            "compiled-autograd",
            {
                **gradient_settings(),
                **compile_settings(compiled_autograd="true"),
            },
            admission_status="passed",
        ),
        {"scale": 3.0},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert events == [("compile", True)]
    assert torch_dynamo_config.compiled_autograd is original_compiled_autograd
    assert torch.allclose(
        tree_leaves(operation())[0],
        torch.tensor([12.0], dtype=torch.float64),
    )
    assert events == [("compile", True), ("call", True)]
    assert torch_dynamo_config.compiled_autograd is original_compiled_autograd


def test_standard_runtime_rejects_compiled_autograd_without_backward_graph() -> None:
    factory = vpx.standard_operation_factory(
        ops.metric(
            "metric",
            "dense",
            aggregation="sum",
            representation=dense_metric_representation(),
        ),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
    )

    with pytest.raises(vp.MaterializationError, match="backward or higher-order"):
        factory(
            vpx.Candidate(
                "metric",
                "compiled-autograd",
                {
                    **metric_settings(),
                    **compile_settings(compiled_autograd="true"),
                },
                admission_status="passed",
            ),
            {"metric_matrix": torch.eye(1, dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_composition_runtime_config_runs_and_materializes_selected_operator(
    tmp_path: Path,
) -> None:
    model = OneParameterModule()
    operator = ops.composition(
        "compose",
        "scale_then_shift",
        aggregation="none",
        children=("multiply", "shift"),
    )
    candidates = (
        vpx.Candidate(
            "compose",
            "row",
            composition_settings(),
            admission_status="passed",
        ),
    )
    runtime = vpx.composition_runtime_config(
        operator,
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
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = vp.materialize(plan, name="compose")
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


@pytest.mark.parametrize(
    (
        "candidate_id",
        "settings",
        "vector",
    ),
    [
        (
            "single-loop-vectors",
            {
                **composition_settings(),
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            _vector_rows("w", COMPOSITION_THREE_VECTOR_ROWS),
        ),
        (
            "manual-batch-vectors",
            {
                **composition_settings(),
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            _vector_rows("w", COMPOSITION_FIVE_VECTOR_ROWS),
        ),
        (
            "vmap-vectors",
            {
                **composition_settings(),
                **torch_func_settings(requires_forward_ad=False),
                "vectorization.mode": "vmap",
                "vectorization.vmap_chunk_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            _vector_rows("w", COMPOSITION_THREE_VECTOR_ROWS),
        ),
    ],
)
def test_composition_vectorization_runs_batched_vectors(
    candidate_id: str,
    settings: Mapping[str, object],
    vector: dict[str, torch.Tensor],
) -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "vectorized",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
    )
    result = factory(
        vpx.Candidate(
            "compose",
            candidate_id,
            settings,
            admission_status="passed",
        ),
        {"scale": 2.0},
        vector,
    )()

    torch.testing.assert_close(tree_leaves(result)[0], vector["w"] * 2.0 + 1.0)


def test_composition_reference_check_uses_anchor_components() -> None:
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "scale_then_shift",
            aggregation="none",
            children=("multiply", "shift"),
        ),
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
            vpx.Candidate(
                "compose",
                "row",
                composition_settings(
                    child_evaluation="selected_child_rows",
                    validation="validate_each_child",
                ),
                admission_status="passed",
            ),
            {"scale": 2.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_reference_check_fails_intermediate_component_mismatch() -> None:
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "hidden_component_error",
            aggregation="none",
            children=("first", "second"),
        ),
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
            vpx.Candidate(
                "compose",
                "row",
                composition_settings(
                    child_evaluation="selected_child_rows",
                    validation="validate_each_child",
                ),
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_reference_check_runs_child_anchor() -> None:
    child = vpx.CompositionChild(
        name="first",
        candidate=vpx.Candidate(
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
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "child_anchor",
            aggregation="none",
            children=("first",),
        ),
        components={"first": child.component},
        anchor_components={"first": child.anchor_component},
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        children=(child,),
    )

    with pytest.raises(vp.ReferenceFailedError, match="child anchor failed"):
        check(
            vpx.Candidate(
                "compose",
                "row",
                composition_settings(
                    child_evaluation="selected_child_rows",
                    validation="validate_each_child",
                ),
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_tune_writes_child_reference_rows(tmp_path: Path) -> None:
    model = OneParameterModule()
    operator = ops.composition(
        "compose",
        "child_anchor",
        aggregation="none",
        children=("identity",),
    )
    child = vpx.CompositionChild(
        name="identity",
        candidate=vpx.Candidate(
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
        vpx.Candidate(
            "compose",
            "row",
            composition_settings(
                child_evaluation="selected_child_rows",
                validation="validate_each_child",
            ),
            admission_status="passed",
        ),
    )
    runtime = vpx.composition_runtime_config(
        operator,
        components={"identity": child.component},
        anchor_components={"identity": child.anchor_component},
        children=(child,),
        candidates=candidates,
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={"identity": "test.identity_component"},
        anchor_component_signature={"identity": "test.identity_component"},
        axis_registry=None,
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = tune_problem(
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
    assert parent_record.measurements["child_reference_rows"] == (
        {
            "child_name": "identity",
            "row": child_record.row_key(),
        },
    )
    assert (
        tmp_path
        / "candidates"
        / child.candidate.family
        / child.candidate.candidate_id
        / "candidate.json"
    ).exists()


@pytest.mark.parametrize(
    "execution",
    ["stream_child_outputs", "materialize_each_child"],
)
def test_composition_selected_child_rows_drive_operation(
    tmp_path: Path,
    execution: str,
) -> None:
    model = OneParameterModule()
    operator = ops.composition(
        "compose",
        "selected_child",
        aggregation="none",
        children=("identity",),
    )
    child = vpx.CompositionChild(
        name="identity",
        candidate=vpx.Candidate(
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
    candidate = vpx.Candidate(
        "compose",
        "row",
        composition_settings(
            execution=execution,
            child_evaluation="selected_child_rows",
            validation="validate_each_child",
        ),
        admission_status="passed",
    )
    runtime = vpx.composition_runtime_config(
        operator,
        components={"identity": wrong_shift_component},
        anchor_components={"identity": identity_component},
        children=(child,),
        candidates=(candidate,),
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={"identity": "test.wrong_shift_component"},
        anchor_component_signature={"identity": "test.identity_component"},
        axis_registry=None,
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    selected = plan.materialize("compose")
    result = selected(
        {"family": "compose", "scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([3.0], dtype=torch.float64),
    )


def test_composition_materialize_each_child_rejects_inline_child_lowering() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "bad_materialized_child",
            aggregation="none",
            children=("identity",),
        ),
        components={"identity": identity_component},
    )

    with pytest.raises(vp.MaterializationError, match="selected_child_rows"):
        factory(
            vpx.Candidate(
                "compose",
                "bad-materialized-child",
                composition_settings(execution="materialize_each_child"),
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_compile_whole_operator_uses_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    recorder = CompileRecorder(events)
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    operator = ops.composition(
        "compose",
        "compiled_composition",
        aggregation="none",
        children=("identity",),
    )
    factory = vpx.composition_operation_factory(
        operator,
        components={"identity": identity_component},
    )
    operation = factory(
        vpx.Candidate(
            "compose",
            "compiled",
            {
                **composition_settings(execution="compile_whole_composition"),
                **compile_settings(),
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )
    output = operation()

    assert events == [("compile", False)]
    assert torch.allclose(
        tree_leaves(output)[0],
        torch.tensor([3.0], dtype=torch.float64),
    )


def test_composition_runs_real_torch_compile_whole_composition() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "compiled_composition",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
    )
    operation = factory(
        vpx.Candidate(
            "compose",
            "compiled",
            {
                **composition_settings(execution="compile_whole_composition"),
                **compile_settings(),
            },
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )
    output = operation()

    torch.testing.assert_close(
        tree_leaves(output)[0],
        torch.tensor([7.0], dtype=torch.float64),
    )


def test_composition_runs_real_torch_compile_child_boundary() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "compiled_children",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
    )
    operation = factory(
        vpx.Candidate(
            "compose",
            "compiled-children",
            {
                **composition_settings(execution="stream_child_outputs"),
                **compile_settings(boundary="composition_child"),
            },
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )
    output = operation()

    torch.testing.assert_close(
        tree_leaves(output)[0],
        torch.tensor([7.0], dtype=torch.float64),
    )


def test_composition_executes_preallocated_output_buffers() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "preallocated_composition",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
    )
    operation = factory(
        vpx.Candidate(
            "compose",
            "preallocated",
            {
                **composition_settings(),
                "memory.output_buffers": "preallocated",
            },
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )
    first = tensor_mapping(operation())
    second = tensor_mapping(operation())

    assert torch.equal(first["w"], torch.tensor([7.0], dtype=torch.float64))
    assert torch.equal(second["w"], torch.tensor([7.0], dtype=torch.float64))
    assert first["w"].data_ptr() == second["w"].data_ptr()


def test_composition_executes_intermediate_residency_between_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    second_inputs = []

    def first_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
        assert "scale" in batch
        vector_map = tensor_mapping(vector)

        return {"w": vector_map["w"] * 2.0}

    def second_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
        assert "scale" in batch
        vector_map = tensor_mapping(vector)
        second_inputs.append(vector_map["w"].data_ptr())

        return {"w": vector_map["w"] + 1.0}

    def recording_residency(
        tensor: torch.Tensor,
        residency: object,
        key: str,
    ) -> torch.Tensor:
        moved = tensor.clone()
        calls.append((key, residency, moved.data_ptr(), moved.detach().clone()))

        return moved

    monkeypatch.setattr(runtime_module, "_residency_tensor", recording_residency)
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "intermediate_residency",
            aggregation="none",
            children=("first", "second"),
        ),
        components={
            "first": first_component,
            "second": second_component,
        },
    )
    result = factory(
        vpx.Candidate(
            "compose",
            "intermediate-residency",
            {
                **composition_settings(),
                "memory.intermediate_residency": "cpu_staged",
            },
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )()
    result_map = tensor_mapping(result)

    assert len(calls) == 1
    assert calls[0][0] == "memory.intermediate_residency"
    assert calls[0][1] == "cpu_staged"
    assert second_inputs == [calls[0][2]]
    torch.testing.assert_close(calls[0][3], torch.tensor([6.0], dtype=torch.float64))
    torch.testing.assert_close(
        result_map["w"],
        torch.tensor([7.0], dtype=torch.float64),
    )


def test_composition_rejects_intermediate_residency_for_fused_children() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "fused_intermediate_residency",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        fused_components={("multiply", "shift"): fused_multiply_shift_component},
    )

    with pytest.raises(vp.MaterializationError, match="visible composition child"):
        factory(
            vpx.Candidate(
                "compose",
                "fused-intermediate-residency",
                {
                    **composition_settings(execution="fuse_adjacent_children"),
                    "memory.intermediate_residency": "cpu_staged",
                },
                admission_status="passed",
            ),
            {"scale": 2.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_compile_whole_operator_requires_compile_axis() -> None:
    registry = vpx.standard_axis_registry()
    candidate = vpx.Candidate(
        "compose",
        "bad",
        composition_settings(execution="compile_whole_composition"),
    )

    admitted = registry.admit(candidate)

    assert admitted.admission_status == "failed"
    assert admitted.admission_error == (
        "compile_whole_composition requires compile.enabled=true"
    )


def test_composition_child_compile_boundary_compiles_each_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    recorder = CompileRecorder(events, called_event="compiled_child")
    monkeypatch.setattr(runtime_module.torch, "compile", recorder.compile)
    patch_runtime_output_recorder(monkeypatch, recorder)
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "compiled_children",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
    )
    operation = factory(
        vpx.Candidate(
            "compose",
            "compiled-children",
            {
                **composition_settings(),
                **compile_settings(boundary="composition_child"),
            },
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert events == [("compile", False), ("compile", False)]
    torch.testing.assert_close(
        tree_leaves(operation())[0],
        torch.tensor([7.0], dtype=torch.float64),
    )
    assert events == [
        ("compile", False),
        ("compile", False),
        ("compiled_child", True),
        ("compiled_child", True),
        ("runtime_output", False),
    ]


def test_composition_child_compile_boundary_rejects_fused_composition() -> None:
    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "compiled_children",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        fused_components={("multiply", "shift"): fused_multiply_shift_component},
    )

    with pytest.raises(vp.MaterializationError, match="child calls"):
        factory(
            vpx.Candidate(
                "compose",
                "compiled-fused",
                {
                    **composition_settings(execution="fuse_adjacent_children"),
                    **compile_settings(boundary="composition_child"),
                },
                admission_status="passed",
            ),
            {"scale": 2.0},
            {"w": torch.tensor([3.0], dtype=torch.float64)},
        )


def test_composition_fuse_adjacent_children_runs_fused_component() -> None:
    events = []

    def child_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
        assert batch["scale"]
        events.append("child")

        return vector

    def fused_component(batch: vpx.Batch, vector: vpx.TensorTree) -> vpx.TensorTree:
        events.append("fused")

        return fused_multiply_shift_component(batch, vector)

    factory = vpx.composition_operation_factory(
        ops.composition(
            "compose",
            "fused",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": child_component,
            "shift": child_component,
        },
        fused_components={("multiply", "shift"): fused_component},
    )
    result = factory(
        vpx.Candidate(
            "compose",
            "fused",
            composition_settings(execution="fuse_adjacent_children"),
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )()

    assert events == ["fused"]
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([7.0], dtype=torch.float64),
    )


def test_composition_fuse_adjacent_children_reference_validates_fused_output() -> None:
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "fused",
            aggregation="none",
            children=("multiply", "shift"),
        ),
        components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        anchor_components={
            "multiply": multiply_component,
            "shift": shift_component,
        },
        fused_components={
            ("multiply", "shift"): fused_multiply_shift_component,
        },
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
    )
    result = check(
        vpx.Candidate(
            "compose",
            "fused",
            composition_settings(execution="fuse_adjacent_children"),
            admission_status="passed",
        ),
        {"scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.measurements["component_errors"] == {
        "multiply": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        "shift": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        "fused_composition": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    }


def test_composition_validate_composed_output_skips_child_anchor() -> None:
    child = vpx.CompositionChild(
        name="identity",
        candidate=vpx.Candidate(
            "child",
            "bad",
            {"operator_path": "child"},
            admission_status="passed",
        ),
        component=identity_component,
        anchor_component=identity_component,
        reference_check=failed_child_reference,
        input_signature={"child": "bad"},
    )
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "composed_only",
            aggregation="none",
            children=("identity",),
        ),
        components={"identity": identity_component},
        anchor_components={"identity": identity_component},
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        children=(child,),
    )
    result = check(
        vpx.Candidate(
            "compose",
            "row",
            composition_settings(
                child_evaluation="selected_child_rows",
                validation="validate_composed_output",
            ),
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert result.child_results == ()


def test_composition_replay_requires_child_reference_rows(tmp_path: Path) -> None:
    model = OneParameterModule()
    operator = ops.composition(
        "compose",
        "child_anchor",
        aggregation="none",
        children=("identity",),
    )
    child = vpx.CompositionChild(
        name="identity",
        candidate=vpx.Candidate(
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
    parent = vpx.Candidate(
        "compose",
        "row",
        composition_settings(
            child_evaluation="selected_child_rows",
            validation="validate_each_child",
        ),
        admission_status="passed",
    )
    runtime = vpx.composition_runtime_config(
        operator,
        components={"identity": child.component},
        anchor_components={"identity": child.anchor_component},
        children=(child,),
        candidates=(parent,),
        thresholds={"max_abs_diff": 1e-9, "max_rel_diff": 1e-9},
        component_signature={"identity": "test.identity_component"},
        anchor_component_signature={"identity": "test.identity_component"},
        axis_registry=None,
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=runtime,
    )
    plan = tune_problem(
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
        / "candidate.json"
    )
    parent_candidate_row = read_record(
        tmp_path / "candidates" / parent.family / parent.candidate_id / "candidate.json"
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")
    full_size_rows = tuple(
        vpx.full_size_record_from_json(
            read_record(
                tmp_path
                / "full_size"
                / record.family
                / record.candidate_id
                / "result.json"
            )
        )
        for record in plan.full_size_records
    )
    check_rows = tuple(
        vpx.check_record_from_json(
            read_record(
                tmp_path
                / "references"
                / record.family
                / record.candidate_id
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

    replayed = vpx.plan_from_json(
        summary,
        replay_context=replay_context_for_plan(plan),
        full_size_records=full_size_rows,
        check_records=check_rows,
        candidate_records=(parent_candidate_row, child_candidate_row),
        materializers=plan.materializers,
        run_dir=tmp_path,
    )
    loaded = vp.load_tuned_plan(
        tmp_path,
        problem,
        memory_backend=CPUMemoryBackend(),
    )

    assert tuple(record.name for record in replayed.check_records) == (
        "child_anchor",
        "composition_anchor",
    )
    assert tuple(record.name for record in loaded.check_records) == (
        "child_anchor",
        "composition_anchor",
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
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
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=full_size_rows,
            check_records=(failed_child, parent_record),
            candidate_records=(parent_candidate_row, child_candidate_row),
            materializers=plan.materializers,
            run_dir=tmp_path,
        )


def test_composition_reference_check_low_precision_anchor() -> None:
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "identity",
            aggregation="none",
            children=("identity",),
        ),
        components={"identity": identity_component},
        anchor_components={"identity": identity_component},
        thresholds={"max_abs_diff": 1e-12, "max_rel_diff": 1e-12},
    )

    with pytest.raises(vp.ReferenceFailedError):
        check(
            vpx.Candidate(
                "compose",
                "float32",
                {**composition_settings(), "dtype.output": "fp32"},
                admission_status="passed",
            ),
            {"scale": 1.0},
            {"w": torch.tensor([1.00000006], dtype=torch.float64)},
        )


def test_composition_reference_check_applies_numeric_error_bound_fields() -> None:
    check = vpx.composition_reference_check(
        ops.composition(
            "compose",
            "identity",
            aggregation="none",
            children=("identity",),
        ),
        components={"identity": identity_component},
        anchor_components={"identity": identity_component},
        thresholds={"max_abs_diff": 1e-4, "max_rel_diff": 1e-3},
        numeric_bound_fields=numeric_bound_fields(),
    )
    result = check(
        vpx.Candidate(
            "compose",
            "high-matmul",
            {
                **composition_settings(),
                "numeric.float32_matmul_precision": "high",
            },
            admission_status="passed",
        ),
        {"scale": 1.0},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert result.measurements["numeric_error_bound_abs"] == pytest.approx(
        1e-8 / (1.0 - 1e-8)
    )


def test_standard_runtime_tunes_hvp_and_materializes_selected_operator(
    tmp_path: Path,
) -> None:
    model = OneParameterModule()
    params = {"w": model.w.detach().clone()}
    candidates = (
        vpx.Candidate(
            "hvp",
            "reverse",
            hvp_settings("reverse_over_reverse"),
            changed_axes=("hvp.path",),
        ),
        vpx.Candidate(
            "hvp",
            "jvp-grad",
            {
                **hvp_settings("jvp_grad"),
                **torch_func_settings(requires_forward_ad=True),
            },
            changed_axes=("hvp.path",),
        ),
    )
    operator = ops.hvp("hvp", "loss", aggregation="sum")
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=ScaleData(),
        operator=operator,
        vectors=ParameterVectorProvider(),
        target=cpu_target(),
        runtime=vpx.standard_runtime_config(
            operator,
            params=params,
            buffers={},
            candidates=candidates,
            thresholds={
                "max_abs_diff": 1e-9,
                "max_rel_diff": 1e-9,
                "directional_abs_diff": 1e-3,
                "directional_rel_diff": 1e-3,
                "symmetry_max_abs_diff": 1e-9,
            },
            objective_signature={"loss": "quadratic-v1"},
            axis_registry=vpx.standard_axis_registry(),
            scalar_objectives={"loss": quadratic_scalar},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0, 2.0, 3.0)),
    )
    selected = vp.materialize(plan, name="hvp")
    result = selected(
        {"family": "hvp", "scale": 2.0},
        {"w": torch.tensor([3.0], dtype=torch.float64)},
    )

    assert plan.selected["hvp"].candidate_id == "jvp-grad"
    assert torch.allclose(
        tree_leaves(result)[0],
        torch.tensor([12.0], dtype=torch.float64),
    )
