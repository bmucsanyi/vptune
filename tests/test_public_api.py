import copy
import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeGuard

import pytest
import torch

import vptune as vp
import vptune.engine.runtime as runtime_module
import vptune.ext as vpx
import vptune.public as public_module
import vptune.tuning.run as run_module
from vptune.core.tensor_tree import tree_leaves


def unchecked_public_value(value: Any) -> Any:
    return value


def test_root_import_surface_exposes_front_door_and_hides_extensions() -> None:
    for name in public_module.__all__:
        assert getattr(vp, name) is getattr(public_module, name)

    for name in (
        "load_plan",
        "load_tuned_plan",
        "load_tuned_run",
        "materialize",
        "tune_run",
        "validate_plan",
    ):
        assert getattr(vp, name) is getattr(run_module, name)

    for name in (
        "AxisDescriptor",
        "AxisManifest",
        "AxisRegistry",
        "Candidate",
        "OperatorSpec",
        "RuntimeConfig",
        "axis_manifest",
        "standard_axis_descriptors",
    ):
        assert not hasattr(vp, name)
        assert hasattr(vpx, name)


class PublicMetricModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[1.0, -2.0], [0.5, 3.0]],
                dtype=torch.float64,
            )
        )

    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"logits": x @ self.weight.T}


class PublicTwoParameterMetricModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        logit_zero = x[:, 0] * self.left[0] + x[:, 1] * self.right[0]
        logit_one = x[:, 0] * self.left[1]

        return {"logits": torch.stack((logit_zero, logit_one), dim=1)}


def typed_metric_model() -> vp.Model:
    module = PublicMetricModule()

    return vp.torch_model(
        module,
        parameters=vp.parameters(module),
        call=vp.module_call(args=("x",), kwargs={}, output="logits"),
    )


def typed_two_parameter_metric_model() -> vp.Model:
    module = PublicTwoParameterMetricModule()

    return vp.torch_model(
        module,
        parameters=vp.parameters(module),
        call=vp.module_call(args=("x",), kwargs={}, output="logits"),
    )


def typed_ekfac_metric() -> vp.Metric:
    basis = torch.eye(2, dtype=torch.float64)
    eigenvalues = torch.tensor(
        [[2.0, 3.0], [5.0, 7.0]],
        dtype=torch.float64,
    )

    return vp.metric.ekfac(
        eigvecs_a={"weight": basis},
        eigvecs_g={"weight": basis},
        corrected_eigenvalues={"weight": eigenvalues},
    )


def typed_vector() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor(
            [[0.5, -1.0], [2.0, 3.0]],
            dtype=torch.float64,
        )
    }


def typed_two_parameter_vector() -> dict[str, torch.Tensor]:
    return {
        "left": torch.tensor([0.5, -1.0], dtype=torch.float64),
        "right": torch.tensor([2.0], dtype=torch.float64),
    }


def declared_psd_matrix_free_output_matvec(
    output: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    _ = output
    diagonal = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=tangent.dtype,
        device=tangent.device,
    )

    return (diagonal * tangent.reshape(-1)).reshape_as(tangent)


def non_callable_value() -> Any:
    return "bad"


def non_combine_value() -> Any:
    return {"child": "curvature"}


def typed_left_vector() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor(
            [[-1.0, 0.25], [0.75, 1.5]],
            dtype=torch.float64,
        )
    }


def typed_ekfac_spectrum() -> torch.Tensor:
    return torch.tensor(
        [[2.0, 3.0], [5.0, 7.0]],
        dtype=torch.float64,
    )


def public_cpu_target() -> vp.Target:
    return vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("fp32", "bf16", "fp16", "fp8_when_supported"),
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=(),
        timing_policy=vp.TimingPolicy(
            short_warmups=0,
            short_measured_calls=1,
            medium_warmups=0,
            medium_measured_calls=1,
            long_warmups=0,
            long_measured_calls=1,
        ),
        selection_policy=vp.SelectionPolicy(),
        determinism_policy=vp.DeterminismPolicy(),
        environment_policy=vp.EnvironmentPolicy(),
    )


def squared_weight_loss(
    params: vpx.ParameterTree,
    buffers: vpx.BufferTree,
    batch: vpx.Batch,
    context: vpx.ObjectiveContext,
) -> torch.Tensor:
    _ = buffers, batch, context

    return params["weight"].square().sum()


def kfac_factor_split(
    left: torch.Tensor,
    right: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = torch.sqrt(left.new_tensor(lam))
    left_mean = torch.trace(left) / left.shape[0]
    right_mean = torch.trace(right) / right.shape[0]
    ratio = torch.sqrt(left_mean / right_mean)

    return ratio * root, root / ratio


def matrix_square_root_product(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    factors = torch.rsqrt(eigenvalues) if inverse else torch.sqrt(eigenvalues)

    return eigenvectors @ (factors * (eigenvectors.T @ vector))


def ce_weight_gradient_rows(
    model: vp.Model,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    logits = batch["x"] @ model.parameter_values["weight"].T
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = torch.nn.functional.one_hot(batch["labels"], num_classes=2).to(
        dtype=torch.float64
    )

    return torch.stack(
        tuple(
            torch.outer(probabilities[index] - one_hot[index], batch["x"][index])
            for index in range(batch["x"].shape[0])
        )
    )


def ce_weight_ggn_reference(
    model: vp.Model,
    batch: Mapping[str, torch.Tensor],
    vector: Mapping[str, torch.Tensor],
    *,
    mask_key: str | None,
) -> torch.Tensor:
    logits = batch["x"] @ model.parameter_values["weight"].T
    probabilities = torch.softmax(logits, dim=-1)
    tangent_logits = batch["x"] @ vector["weight"].T
    hessian_tangent = probabilities * (
        tangent_logits - (probabilities * tangent_logits).sum(dim=-1, keepdim=True)
    )

    if mask_key is None:
        denominator = logits.new_tensor(batch["labels"].numel())
    else:
        mask = batch[mask_key].to(dtype=logits.dtype)
        hessian_tangent = hessian_tangent * mask[:, None]
        denominator = mask.sum()

    return (hessian_tangent / denominator).T @ batch["x"]


def flat_weight(tree: vpx.TensorTree) -> torch.Tensor:
    return tensor_mapping(tree)["weight"].reshape(-1)


def flat_tree(tree: vpx.TensorTree) -> torch.Tensor:
    return torch.cat(tuple(leaf.reshape(-1) for leaf in tree_leaves(tree)))


def tree_basis_vector(
    template: vpx.TensorTree,
    index: int,
) -> vpx.TensorTree:
    leaves = tree_leaves(template)
    output_leaves = []
    offset = 0

    for leaf in leaves:
        flat = torch.zeros_like(leaf).reshape(-1)
        next_offset = offset + flat.numel()

        if offset <= index < next_offset:
            flat[index - offset] = 1.0

        output_leaves.append(flat.reshape_as(leaf))
        offset = next_offset

    assert offset > index

    if isinstance(template, dict):
        return dict(zip(template, output_leaves, strict=True))

    if isinstance(template, tuple):
        return tuple(output_leaves)

    return output_leaves[0]


def weight_basis_vector(
    template: vpx.TensorTree,
    index: int,
) -> dict[str, torch.Tensor]:
    template_mapping = tensor_mapping(template)
    flat = torch.zeros_like(template_mapping["weight"]).reshape(-1)
    flat[index] = 1.0

    return {"weight": flat.reshape_as(template_mapping["weight"])}


def dense_weight_operator_matrix(
    operator: vp.Operator,
    batch: vpx.Batch,
    template: vpx.TensorTree,
) -> torch.Tensor:
    columns = tuple(
        flat_weight(operator(batch, weight_basis_vector(template, index)))
        for index in range(flat_weight(template).numel())
    )

    return torch.stack(columns, dim=1)


def dense_tree_operator_matrix(
    operator: vp.Operator,
    batch: vpx.Batch,
    template: vpx.TensorTree,
) -> torch.Tensor:
    columns = tuple(
        flat_tree(operator(batch, tree_basis_vector(template, index)))
        for index in range(flat_tree(template).numel())
    )

    return torch.stack(columns, dim=1)


def dense_weight_unary_operator_matrix(
    operator: vp.Operator,
    template: vpx.TensorTree,
) -> torch.Tensor:
    columns = tuple(
        flat_weight(operator(weight_basis_vector(template, index)))
        for index in range(flat_weight(template).numel())
    )

    return torch.stack(columns, dim=1)


def dense_tree_unary_operator_matrix(
    operator: vp.Operator,
    template: vpx.TensorTree,
) -> torch.Tensor:
    columns = tuple(
        flat_tree(operator(tree_basis_vector(template, index)))
        for index in range(flat_tree(template).numel())
    )

    return torch.stack(columns, dim=1)


def assert_dense_weight_metric_products(
    model: vp.Model,
    metric: vp.Metric,
    matrix: torch.Tensor,
    *,
    damping: float,
) -> None:
    vector = typed_vector()
    left = typed_left_vector()
    flat_vector = flat_weight(vector)
    flat_left = flat_weight(left)
    damped = matrix + damping * torch.eye(
        matrix.shape[0],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    expected_metric = matrix @ flat_vector
    expected_inverse = torch.linalg.solve(damped, flat_vector)

    metric_product = vp.metric_vp(model, metric)(vector)
    inverse_product = vp.inverse_metric_vp(
        model,
        metric,
        damping=vp.damping.scalar(damping),
    )(vector)
    metric_inner = vp.metric_inner_vp(model, metric)(left, vector)
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        metric,
        damping=vp.damping.scalar(damping),
    )(left, vector)

    torch.testing.assert_close(flat_weight(metric_product), expected_metric)
    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(metric_inner, flat_left @ expected_metric)
    torch.testing.assert_close(inverse_inner, flat_left @ expected_inverse)


def is_tensor_mapping(tree: vpx.TensorTree) -> TypeGuard[dict[str, torch.Tensor]]:
    if not isinstance(tree, dict):
        return False

    return all(isinstance(value, torch.Tensor) for value in tree.values())


def tensor_mapping(tree: vpx.TensorTree) -> dict[str, torch.Tensor]:
    assert is_tensor_mapping(tree)

    return tree


def public_replay_context_for_plan(plan: vpx.Plan) -> vpx.ReplayContext:
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
        validation_required=plan.validation_required,
        validation_order=plan.validation_order,
    )


def replay_context_with_stale_composition_coefficient(
    context: vpx.ReplayContext,
    family: str,
    coefficient: float,
) -> vpx.ReplayContext:
    family_input_signatures = copy.deepcopy(dict(context.family_input_signatures))
    operator = family_input_signatures[family]["operator"]
    semantics = operator["semantics"]
    combine = semantics["combine"]
    terms = combine["terms"]
    terms[0]["coefficient"] = coefficient

    return dataclasses.replace(
        context,
        family_input_signatures=family_input_signatures,
    )


def test_typed_ekfac_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    metric = typed_ekfac_metric()
    vector = typed_vector()
    spectrum = typed_ekfac_spectrum()
    lam = 0.25

    metric_product = vp.metric_vp(model, metric)(vector)
    inverse_product = vp.inverse_metric_vp(
        model,
        metric,
        damping=vp.damping.eigenvalue_floor(lam),
    )(vector)
    square_root_product = vp.sqrt_metric_vp(model, metric)(vector)
    inverse_square_root_product = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=vp.damping.eigenvalue_floor(lam),
    )(vector)

    torch.testing.assert_close(
        metric_product["weight"],
        spectrum * vector["weight"],
    )
    torch.testing.assert_close(
        inverse_product["weight"],
        vector["weight"] / (spectrum + lam),
    )
    torch.testing.assert_close(
        square_root_product["weight"],
        torch.sqrt(spectrum) * vector["weight"],
    )
    torch.testing.assert_close(
        inverse_square_root_product["weight"],
        torch.rsqrt(spectrum + lam) * vector["weight"],
    )


def test_typed_ekfac_metric_inner_products_execute_against_reference() -> None:
    model = typed_metric_model()
    metric = typed_ekfac_metric()
    left = typed_left_vector()
    right = typed_vector()
    spectrum = typed_ekfac_spectrum()
    lam = 0.25

    metric_inner = vp.metric_inner_vp(model, metric)(left, right)
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        metric,
        damping=vp.damping.eigenvalue_floor(lam),
    )(left, right)

    torch.testing.assert_close(
        metric_inner,
        (left["weight"] * spectrum * right["weight"]).sum(),
    )
    torch.testing.assert_close(
        inverse_inner,
        (left["weight"] * right["weight"] / (spectrum + lam)).sum(),
    )


def test_typed_dense_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    matrix = torch.diag(torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64))
    metric = vp.metric.dense(matrix=matrix)

    assert_dense_weight_metric_products(model, metric, matrix, damping=0.25)


def test_typed_diagonal_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    diagonal = {
        "weight": torch.tensor(
            [[2.0, 3.0], [5.0, 7.0]],
            dtype=torch.float64,
        )
    }
    metric = vp.metric.diagonal(diag=diagonal)
    matrix = torch.diag(diagonal["weight"].reshape(-1))

    assert_dense_weight_metric_products(model, metric, matrix, damping=0.25)


def test_typed_diagonal_metric_per_group_damping_executes_by_parameter() -> None:
    model = typed_metric_model()
    diagonal = {
        "weight": torch.tensor(
            [[2.0, 3.0], [5.0, 7.0]],
            dtype=torch.float64,
        )
    }
    metric = vp.metric.diagonal(diag=diagonal)
    damping = vp.damping.per_group({"weight": 0.25})
    vector = typed_vector()
    left = typed_left_vector()
    denominator = diagonal["weight"] + 0.25

    inverse_operator = vp.inverse_metric_vp(model, metric, damping=damping)
    inverse_inner_operator = vp.inverse_metric_inner_vp(
        model,
        metric,
        damping=damping,
    )
    inverse_sqrt_operator = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=damping,
    )

    inverse_product = inverse_operator(vector)
    inverse_inner = inverse_inner_operator(left, vector)
    inverse_square_root = inverse_sqrt_operator(vector)

    assert inverse_operator.spec.semantics["damping_kind"] == "per_group"
    assert inverse_operator.spec.semantics["damping"] == {"weight": 0.25}
    torch.testing.assert_close(
        inverse_product["weight"],
        vector["weight"] / denominator,
    )
    torch.testing.assert_close(
        inverse_inner,
        (left["weight"] * vector["weight"] / denominator).sum(),
    )
    torch.testing.assert_close(
        inverse_square_root["weight"],
        torch.rsqrt(denominator) * vector["weight"],
    )


def test_typed_block_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    blocks = {
        "first": torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64),
        "second": torch.tensor([[4.0, 1.0], [1.0, 5.0]], dtype=torch.float64),
    }
    matrix = torch.block_diag(blocks["first"], blocks["second"])
    metric = vp.metric.block_diagonal(blocks=blocks)

    assert_dense_weight_metric_products(model, metric, matrix, damping=0.25)


def test_typed_block_metric_per_group_damping_executes_by_block() -> None:
    model = typed_metric_model()
    blocks = {
        "first": torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64),
        "second": torch.tensor([[4.0, 1.0], [1.0, 5.0]], dtype=torch.float64),
    }
    metric = vp.metric.block_diagonal(blocks=blocks)
    damping = vp.damping.per_group({"first": 0.25, "second": 0.75})
    vector = typed_vector()
    left = typed_left_vector()
    flat_vector = vector["weight"].reshape(-1)
    flat_left = left["weight"].reshape(-1)
    damped_matrix = torch.block_diag(
        blocks["first"] + 0.25 * torch.eye(2, dtype=torch.float64),
        blocks["second"] + 0.75 * torch.eye(2, dtype=torch.float64),
    )
    inverse_operator = vp.inverse_metric_vp(model, metric, damping=damping)
    inverse_inner_operator = vp.inverse_metric_inner_vp(
        model,
        metric,
        damping=damping,
    )
    inverse_sqrt_operator = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=damping,
    )

    inverse_product = inverse_operator(vector)
    inverse_inner = inverse_inner_operator(left, vector)
    inverse_square_root = inverse_sqrt_operator(vector)
    expected_inverse = torch.linalg.solve(damped_matrix, flat_vector)
    expected_inverse_square_root = (
        torch.linalg.cholesky(torch.linalg.inv(damped_matrix)) @ flat_vector
    )

    assert inverse_operator.spec.semantics["damping_kind"] == "per_group"
    assert inverse_operator.spec.semantics["damping"] == {
        "first": 0.25,
        "second": 0.75,
    }
    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(inverse_inner, flat_left @ expected_inverse)
    torch.testing.assert_close(
        flat_weight(inverse_square_root),
        expected_inverse_square_root,
    )


def test_typed_low_rank_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    factor = torch.tensor(
        [[1.0], [2.0], [-1.0], [0.5]],
        dtype=torch.float64,
    )
    diagonal = torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    matrix = factor @ factor.T + torch.diag(diagonal)
    metric = vp.metric.low_rank(factor=factor, diagonal=diagonal)

    assert_dense_weight_metric_products(model, metric, matrix, damping=0.25)


def test_typed_low_rank_metric_square_root_accepts_declared_latent_width() -> None:
    model = typed_metric_model()
    factor = torch.tensor(
        [[1.0], [2.0], [-1.0], [0.5]],
        dtype=torch.float64,
    )
    diagonal = torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    metric = vp.metric.low_rank(factor=factor, diagonal=diagonal)
    latent = torch.tensor([0.25, -1.0, 0.5, 2.0, -0.75], dtype=torch.float64)
    rank = factor.shape[1]
    expected = factor @ latent[:rank] + torch.sqrt(diagonal) * latent[rank:]

    square_root_product = vp.sqrt_metric_vp(model, metric)(latent)

    torch.testing.assert_close(flat_weight(square_root_product), expected)


def test_typed_low_rank_inverse_square_root_is_declared_factor() -> None:
    model = typed_metric_model()
    factor = torch.tensor(
        [[1.0], [2.0], [-1.0], [0.5]],
        dtype=torch.float64,
    )
    diagonal = torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    metric = vp.metric.low_rank(factor=factor, diagonal=diagonal)
    vector = typed_vector()
    damping = 0.25
    damped_matrix = factor @ factor.T + torch.diag(diagonal)
    damped_matrix = damped_matrix + damping * torch.eye(4, dtype=torch.float64)
    operator = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=vp.damping.scalar(damping),
    )

    factor_matrix = dense_weight_unary_operator_matrix(operator, vector)

    torch.testing.assert_close(
        factor_matrix @ factor_matrix.T,
        torch.linalg.inv(damped_matrix),
        atol=1e-10,
        rtol=1e-10,
    )


def test_typed_kfac_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    left_factor = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    right_factor = torch.tensor([[4.0, 1.0], [1.0, 5.0]], dtype=torch.float64)
    matrix = torch.kron(left_factor, right_factor)
    metric = vp.metric.kfac(
        factors={
            "weight": {
                "left_factor": left_factor,
                "right_factor": right_factor,
            }
        }
    )
    vector = typed_vector()
    left = typed_left_vector()
    flat_vector = vector["weight"].reshape(-1)
    flat_left = left["weight"].reshape(-1)
    lam = 0.25

    metric_product = vp.metric_vp(model, metric)(vector)
    inverse_product = vp.inverse_metric_vp(
        model,
        metric,
        damping=vp.damping.scalar(lam),
    )(vector)
    square_root_product = vp.sqrt_metric_vp(model, metric)(vector)
    inverse_square_root_product = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=vp.damping.scalar(lam),
    )(vector)
    metric_inner = vp.metric_inner_vp(model, metric)(left, vector)
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        metric,
        damping=vp.damping.scalar(lam),
    )(left, vector)

    damped_matrix = matrix + lam * torch.eye(4, dtype=torch.float64)
    expected_metric = matrix @ flat_vector
    expected_inverse = torch.linalg.solve(damped_matrix, flat_vector)
    expected_square_root = matrix_square_root_product(matrix, flat_vector)
    expected_inverse_square_root = matrix_square_root_product(
        damped_matrix,
        flat_vector,
        inverse=True,
    )

    torch.testing.assert_close(flat_weight(metric_product), expected_metric)
    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(flat_weight(square_root_product), expected_square_root)
    torch.testing.assert_close(
        flat_weight(inverse_square_root_product),
        expected_inverse_square_root,
    )
    torch.testing.assert_close(metric_inner, flat_left @ expected_metric)
    torch.testing.assert_close(inverse_inner, flat_left @ expected_inverse)


def test_typed_kfac_metric_per_group_damping_executes_by_parameter() -> None:
    model = typed_metric_model()
    left_factor = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    right_factor = torch.tensor([[4.0, 1.0], [1.0, 5.0]], dtype=torch.float64)
    metric = vp.metric.kfac(
        factors={
            "weight": {
                "left_factor": left_factor,
                "right_factor": right_factor,
            }
        }
    )
    damping = vp.damping.per_group({"weight": 0.25})
    vector = typed_vector()
    left = typed_left_vector()
    flat_vector = vector["weight"].reshape(-1)
    flat_left = left["weight"].reshape(-1)
    damped_matrix = torch.kron(left_factor, right_factor) + 0.25 * torch.eye(
        4,
        dtype=torch.float64,
    )

    inverse_product = vp.inverse_metric_vp(model, metric, damping=damping)(vector)
    inverse_inner = vp.inverse_metric_inner_vp(model, metric, damping=damping)(
        left,
        vector,
    )
    inverse_square_root_product = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=damping,
    )(vector)

    expected_inverse = torch.linalg.solve(damped_matrix, flat_vector)
    expected_inverse_square_root = matrix_square_root_product(
        damped_matrix,
        flat_vector,
        inverse=True,
    )

    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(inverse_inner, flat_left @ expected_inverse)
    torch.testing.assert_close(
        flat_weight(inverse_square_root_product),
        expected_inverse_square_root,
    )


def test_typed_kfac_pi_damping_uses_factored_shift() -> None:
    model = typed_metric_model()
    left_factor = torch.tensor([[5.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    right_factor = torch.tensor([[3.0, 0.25], [0.25, 1.5]], dtype=torch.float64)
    metric = vp.metric.kfac(
        factors={
            "weight": {
                "left_factor": left_factor,
                "right_factor": right_factor,
            }
        }
    )
    vector = typed_vector()
    flat_vector = vector["weight"].reshape(-1)
    lam = 0.25
    left_shift, right_shift = kfac_factor_split(left_factor, right_factor, lam)
    left_damped = left_factor + left_shift * torch.eye(2, dtype=torch.float64)
    right_damped = right_factor + right_shift * torch.eye(2, dtype=torch.float64)
    inverse_operator = vp.inverse_metric_vp(
        model,
        metric,
        damping=vp.damping.kfac_pi(lam),
    )
    inverse_sqrt_operator = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=vp.damping.kfac_pi(lam),
    )

    inverse_product = inverse_operator(vector)
    inverse_sqrt_product = inverse_sqrt_operator(vector)
    left_solved = torch.linalg.solve(left_damped, vector["weight"])
    expected_inverse = torch.linalg.solve(right_damped, left_solved.T).T.reshape(-1)
    expected_inverse_sqrt = matrix_square_root_product(
        torch.kron(left_damped, right_damped),
        flat_vector,
        inverse=True,
    )
    scalar_exact = torch.linalg.solve(
        torch.kron(left_factor, right_factor) + lam * torch.eye(4, dtype=torch.float64),
        flat_vector,
    )

    assert inverse_operator.spec.semantics["damping_kind"] == "kfac_pi"
    assert inverse_operator.spec.semantics["damping_policy"] == "trace_norm"
    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(
        flat_weight(inverse_sqrt_product),
        expected_inverse_sqrt,
    )
    assert not torch.allclose(flat_weight(inverse_product), scalar_exact)


def test_typed_ekfac_metric_per_group_damping_executes_by_block() -> None:
    model = typed_metric_model()
    metric = typed_ekfac_metric()
    damping = vp.damping.per_group({"weight": 0.25})
    vector = typed_vector()
    left = typed_left_vector()
    flat_vector = flat_weight(vector)
    flat_left = flat_weight(left)
    damped_spectrum = typed_ekfac_spectrum().reshape(-1) + 0.25

    inverse_operator = vp.inverse_metric_vp(model, metric, damping=damping)
    inverse_product = inverse_operator(vector)
    inverse_inner = vp.inverse_metric_inner_vp(model, metric, damping=damping)(
        left,
        vector,
    )
    inverse_sqrt_product = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=damping,
    )(vector)

    expected_inverse = flat_vector / damped_spectrum
    expected_inverse_square_root = flat_vector / torch.sqrt(damped_spectrum)

    assert inverse_operator.spec.semantics["damping_kind"] == "per_group"
    assert inverse_operator.spec.semantics["damping_value"] == {"weight": 0.25}
    torch.testing.assert_close(flat_weight(inverse_product), expected_inverse)
    torch.testing.assert_close(inverse_inner, flat_left @ expected_inverse)
    torch.testing.assert_close(
        flat_weight(inverse_sqrt_product),
        expected_inverse_square_root,
    )


def test_typed_ggn_derived_metric_products_execute_against_reference() -> None:
    model = typed_metric_model()
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.5, 1.0, -0.5, 0.25],
            [0.0, -1.0, 1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    loss_hessian = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
    matrix = jacobian.T @ loss_hessian @ jacobian
    metric = vp.metric.ggn_derived(
        factors={"jacobian": jacobian, "loss_hessian": loss_hessian}
    )

    assert_dense_weight_metric_products(model, metric, matrix, damping=0.25)


def test_typed_ggn_derived_metric_square_root_accepts_output_latent_width() -> None:
    model = typed_metric_model()
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.5, 1.0, -0.5, 0.25],
            [0.0, -1.0, 1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    loss_hessian = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
    metric = vp.metric.ggn_derived(
        factors={"jacobian": jacobian, "loss_hessian": loss_hessian}
    )
    latent = torch.tensor([0.25, -1.0, 0.5], dtype=torch.float64)
    loss_root = torch.diag(torch.sqrt(torch.diag(loss_hessian)))
    expected = jacobian.T @ (loss_root @ latent)

    square_root_product = vp.sqrt_metric_vp(model, metric)(latent)

    torch.testing.assert_close(flat_weight(square_root_product), expected)


def test_typed_ggn_derived_inverse_square_root_is_declared_factor() -> None:
    model = typed_metric_model()
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.5, 1.0, -0.5, 0.25],
            [0.0, -1.0, 1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    loss_hessian = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
    metric = vp.metric.ggn_derived(
        factors={"jacobian": jacobian, "loss_hessian": loss_hessian}
    )
    vector = typed_vector()
    damping = 0.25
    damped_matrix = jacobian.T @ loss_hessian @ jacobian
    damped_matrix = damped_matrix + damping * torch.eye(4, dtype=torch.float64)
    operator = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        damping=vp.damping.scalar(damping),
    )

    factor_matrix = dense_weight_unary_operator_matrix(operator, vector)

    torch.testing.assert_close(
        factor_matrix @ factor_matrix.T,
        torch.linalg.inv(damped_matrix),
        atol=1e-10,
        rtol=1e-10,
    )


def test_typed_parameter_surface_per_group_damping_executes() -> None:
    model = typed_two_parameter_metric_model()
    vector = typed_two_parameter_vector()
    left = {
        "left": torch.tensor([-1.5, 0.75], dtype=torch.float64),
        "right": torch.tensor([0.25], dtype=torch.float64),
    }
    flat_vector = flat_tree(vector)
    flat_left = flat_tree(left)
    damping = vp.damping.per_group({"left": 0.25, "right": 0.75})
    damping_diagonal = torch.tensor([0.25, 0.25, 0.75], dtype=torch.float64)
    dense_matrix = torch.tensor(
        [
            [4.0, 0.25, -0.5],
            [0.25, 3.0, 0.75],
            [-0.5, 0.75, 2.5],
        ],
        dtype=torch.float64,
    )
    low_rank_factor = torch.tensor([[1.0], [-0.5], [0.25]], dtype=torch.float64)
    low_rank_diagonal = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    low_rank_matrix = low_rank_factor @ low_rank_factor.T
    low_rank_matrix = low_rank_matrix + torch.diag(low_rank_diagonal)
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.25, 1.0, -0.5],
            [0.5, -0.25, 1.0],
        ],
        dtype=torch.float64,
    )
    loss_hessian = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    ggn_matrix = jacobian.T @ loss_hessian @ jacobian
    cases = (
        ("dense", vp.metric.dense(matrix=dense_matrix), dense_matrix),
        (
            "low_rank",
            vp.metric.low_rank(factor=low_rank_factor, diagonal=low_rank_diagonal),
            low_rank_matrix,
        ),
        (
            "ggn",
            vp.metric.ggn_derived(
                factors={"jacobian": jacobian, "loss_hessian": loss_hessian}
            ),
            ggn_matrix,
        ),
    )

    for _, metric, matrix in cases:
        damped = matrix + torch.diag(damping_diagonal)
        inverse_operator = vp.inverse_metric_vp(model, metric, damping=damping)
        inverse_inner_operator = vp.inverse_metric_inner_vp(
            model,
            metric,
            damping=damping,
        )
        inverse_sqrt_operator = vp.inverse_sqrt_metric_vp(
            model,
            metric,
            damping=damping,
        )
        inverse_product = inverse_operator(vector)
        inverse_inner = inverse_inner_operator(left, vector)
        factor_matrix = dense_tree_unary_operator_matrix(
            inverse_sqrt_operator,
            vector,
        )
        expected_inverse = torch.linalg.solve(damped, flat_vector)

        torch.testing.assert_close(
            flat_tree(inverse_product),
            expected_inverse,
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            inverse_inner,
            flat_left @ expected_inverse,
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            factor_matrix @ factor_matrix.T,
            torch.linalg.inv(damped),
            atol=1e-10,
            rtol=1e-10,
        )
        assert inverse_operator.spec.semantics["damping_kind"] == "per_group"
        assert inverse_operator.spec.semantics["damping_groups"] == (
            {"name": "left", "start": 0, "stop": 2},
            {"name": "right", "start": 2, "stop": 3},
        )


def test_typed_matrix_free_metric_per_group_damping_uses_parameter_surface(
    tmp_path: Path,
) -> None:
    model = typed_two_parameter_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1, 0], dtype=torch.long),
        "symmetry_vector": typed_two_parameter_vector(),
    }
    vector = typed_two_parameter_vector()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    curvature = vp.ggnvp(model, loss, name="two_parameter_curvature")
    metric = vp.metric.matrix_free(operator=curvature)
    damping = vp.damping.per_group({"left": 0.25, "right": 0.75})
    inverse_product = vp.inverse_metric_vp(
        model,
        metric,
        name="two_parameter_inverse",
        damping=damping,
    )
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        metric,
        name="two_parameter_inverse_inner",
        damping=damping,
    )
    run = vp.tune(
        products=(curvature, inverse_product, inverse_inner),
        model=model,
        data={
            "two_parameter_curvature": (batch,),
            "two_parameter_inverse": (batch,),
            "two_parameter_inverse_inner": (batch,),
        },
        vectors={
            "two_parameter_curvature": (vector,),
            "two_parameter_inverse": (vector,),
            "two_parameter_inverse_inner": ((batch["symmetry_vector"], vector),),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    dense_matrix = dense_tree_operator_matrix(curvature, batch, vector)
    damping_diagonal = torch.tensor([0.25, 0.25, 0.75], dtype=torch.float64)
    expected = torch.linalg.solve(
        dense_matrix + torch.diag(damping_diagonal),
        flat_tree(vector),
    )
    output = run["two_parameter_inverse"](batch, vector)
    inner_output = run["two_parameter_inverse_inner"](
        batch,
        batch["symmetry_vector"],
        vector,
    )
    flat_left = flat_tree(batch["symmetry_vector"])

    torch.testing.assert_close(flat_tree(output), expected, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(inner_output, flat_left @ expected, atol=1e-8, rtol=1e-8)
    assert inverse_product.spec.semantics["damping_groups"] == (
        {"name": "left", "start": 0, "stop": 2},
        {"name": "right", "start": 2, "stop": 3},
    )


def test_eigenvalue_floor_damping_requires_ekfac_metric() -> None:
    model = typed_metric_model()
    matrix = torch.eye(4, dtype=torch.float64)
    metric = vp.metric.dense(matrix=matrix)

    with pytest.raises(vp.MaterializationError, match="EKFAC"):
        vp.inverse_metric_vp(
            model,
            metric,
            damping=vp.damping.eigenvalue_floor(0.1),
        )


def test_per_group_damping_requires_named_block_metric_groups() -> None:
    model = typed_metric_model()
    matrix = torch.eye(4, dtype=torch.float64)
    dense_metric = vp.metric.dense(matrix=matrix)
    diagonal_metric = vp.metric.diagonal(diag={"weight": torch.ones((2, 2))})
    block_metric = vp.metric.block_diagonal(
        blocks={
            "first": torch.eye(2, dtype=torch.float64),
            "second": torch.eye(2, dtype=torch.float64),
        }
    )

    with pytest.raises(vp.MaterializationError, match="keys must match"):
        vp.inverse_metric_vp(
            model,
            dense_metric,
            damping=vp.damping.per_group({"other": 0.1}),
        )

    with pytest.raises(vp.MaterializationError, match="keys must match"):
        vp.inverse_metric_vp(
            model,
            diagonal_metric,
            damping=vp.damping.per_group({"other": 0.1}),
        )

    with pytest.raises(vp.MaterializationError, match="keys must match"):
        vp.inverse_metric_vp(
            model,
            block_metric,
            damping=vp.damping.per_group({"first": 0.1}),
        )


def test_kfac_pi_damping_requires_kfac_metric() -> None:
    model = typed_metric_model()
    matrix = torch.eye(4, dtype=torch.float64)
    metric = vp.metric.dense(matrix=matrix)

    with pytest.raises(vp.MaterializationError, match="KFAC"):
        vp.inverse_metric_vp(
            model,
            metric,
            damping=vp.damping.kfac_pi(0.1),
        )


def test_typed_kfac_rejects_invalid_factor_declarations() -> None:
    factor = torch.eye(2, dtype=torch.float64)

    with pytest.raises(vp.MaterializationError, match="left_factor"):
        vp.metric.kfac(factors={"weight": {"right_factor": factor}})

    with pytest.raises(vp.MaterializationError, match="dampings"):
        vp.metric.kfac(
            factors={"weight": {"left_factor": factor, "right_factor": factor}},
            dampings={"weight": 0.1},
        )


def test_typed_inverse_metric_tol_lowers_to_matrix_free_cg() -> None:
    model = typed_metric_model()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    curvature = vp.ggnvp(model, loss, name="curvature")
    matrix_free_metric = vp.metric.matrix_free(operator=curvature)
    inverse = vp.inverse_metric_vp(
        model,
        matrix_free_metric,
        damping=vp.damping.scalar(0.5),
        tol=1e-4,
    )
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        matrix_free_metric,
        damping=vp.damping.scalar(0.5),
        tol=2e-4,
    )

    assert inverse.spec.semantics["tol"] == pytest.approx(1e-4)
    assert inverse.default_settings["inverse_metric.solve_path"] == "conjugate_gradient"
    assert inverse_inner.spec.semantics["tol"] == pytest.approx(2e-4)
    assert (
        inverse_inner.default_settings["inverse_metric.solve_path"]
        == "conjugate_gradient"
    )


def test_typed_inverse_metric_tol_rejects_non_iterative_public_paths() -> None:
    model = typed_metric_model()
    metric = typed_ekfac_metric()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    curvature = vp.ggnvp(model, loss, name="curvature")
    matrix_free_metric = vp.metric.matrix_free(operator=curvature)

    with pytest.raises(vp.MaterializationError, match="matrix_free"):
        vp.inverse_metric_vp(
            model,
            metric,
            damping=vp.damping.eigenvalue_floor(0.1),
            tol=1e-4,
        )

    inverse_sqrt = vp.inverse_sqrt_metric_vp(
        model,
        matrix_free_metric,
        damping=vp.damping.scalar(0.5),
        tol=1e-4,
    )

    assert inverse_sqrt.spec.semantics["tol"] == pytest.approx(1e-4)
    assert inverse_sqrt.default_settings["sqrt_metric.factor_path"] == (
        "matrix_free_lanczos"
    )

    with pytest.raises(vp.MaterializationError, match="matrix_free Lanczos"):
        vp.inverse_sqrt_metric_vp(
            model,
            metric,
            damping=vp.damping.eigenvalue_floor(0.1),
            tol=1e-4,
        )


def test_typed_composition_records_sequential_combine_and_runtime_order() -> None:
    model = typed_metric_model()
    combine = vp.compose("preconditioner", "curvature")
    operator = vp.composition(
        model,
        name="preconditioned_curvature",
        children=("curvature", "preconditioner"),
        combine=combine,
    )

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "composition"
    assert operator.spec.family == "preconditioned_curvature"
    assert operator.spec.semantics["children"] == ("curvature", "preconditioner")
    assert operator.spec.semantics["public_children"] == (
        "curvature",
        "preconditioner",
    )
    assert operator.spec.semantics["combine"] == {
        "kind": "compose",
        "terms": (
            {"kind": "child", "name": "preconditioner"},
            {"kind": "child", "name": "curvature"},
        ),
    }

    with pytest.raises(vp.MaterializationError, match="selected child rows"):
        operator(
            {"x": torch.zeros(1, 2, dtype=torch.float64)},
            typed_vector(),
        )


def test_public_single_product_entrypoints_reject_composition(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    operator = vp.composition(
        model,
        name="preconditioned_curvature",
        children=("curvature", "preconditioner"),
        combine=vp.compose("preconditioner", "curvature"),
    )
    batch = {}
    vector = typed_vector()
    target = public_cpu_target()
    space = vp.space.standard()
    search = vp.search.exhaustive()

    with pytest.raises(
        vp.MaterializationError,
        match=r"operator[.]tune rejects composition",
    ):
        operator.tune(
            data=(batch,),
            vectors=(vector,),
            target=target,
            space=space,
            search=search,
            run_dir=tmp_path / "operator",
        )

    with pytest.raises(
        vp.MaterializationError,
        match=r"vp[.]problem rejects composition",
    ):
        vp.problem(
            operator,
            data=(batch,),
            vectors=(vector,),
            target=target,
            space=space,
            search=search,
        )

    lower_problem = public_module._composition_problem(
        operator,
        data=(batch,),
        vectors=(vector,),
        target=target,
        space=space,
        search=search,
        reference=None,
        probes=None,
    )

    with pytest.raises(
        vp.MaterializationError,
        match=r"vp[.]autotune rejects composition",
    ):
        vp.autotune(lower_problem, run_dir=tmp_path / "problem")


def test_typed_composition_validates_combine_children_and_source_positions() -> None:
    model = typed_metric_model()

    with pytest.raises(vp.MaterializationError, match="leaves must match"):
        vp.composition(
            model,
            children=("curvature", "preconditioner"),
            combine=vp.compose("preconditioner"),
        )

    with pytest.raises(vp.MaterializationError, match="innermost"):
        vp.compose(vp.source("gradient"), "curvature")

    with pytest.raises(vp.MaterializationError, match="source-seeded"):
        vp.composition(
            model,
            children=("gradient", "curvature"),
            combine=vp.linear_combination(
                (1.0, vp.source("gradient")),
                (0.25, "curvature"),
            ),
        )


def test_typed_composition_validates_public_combinator_inputs() -> None:
    model = typed_metric_model()

    with pytest.raises(vp.MaterializationError, match="compose requires"):
        vp.compose()

    with pytest.raises(vp.MaterializationError, match="linear_combination requires"):
        vp.linear_combination()

    with pytest.raises(vp.MaterializationError, match="coefficient-expression pairs"):
        vp.linear_combination(non_callable_value())

    with pytest.raises(vp.MaterializationError, match="coefficient"):
        vp.linear_combination((non_callable_value(), "curvature"))

    with pytest.raises(vp.MaterializationError, match="coefficient"):
        vp.scaled_identity(non_callable_value())

    with pytest.raises(vp.MaterializationError, match="unsupported"):
        vp.compose(non_combine_value())

    with pytest.raises(vp.MaterializationError, match="source child"):
        vp.source("")

    with pytest.raises(vp.MaterializationError, match="children must be nonempty"):
        vp.composition(model, children=(), combine="curvature")

    with pytest.raises(vp.MaterializationError, match="children must be unique"):
        vp.composition(
            model,
            children=("curvature", "curvature"),
            combine="curvature",
        )


def test_typed_loss_from_scalar_gradient_and_hvp_execute_reference() -> None:
    model = typed_metric_model()
    scale = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    batch = {"scale": scale}
    vector = typed_vector()

    def quadratic(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family in {"gradient", "hvp"}

        return 0.5 * (params["weight"] * batch["scale"]).square().sum()

    loss = vp.loss.from_scalar(quadratic, output="logits", version="quadratic-v1")
    gradient = vp.gradient(model, loss)
    hvp = vp.hvp(model, loss)

    gradient_output = gradient(batch)
    hvp_output = hvp(batch, vector)

    torch.testing.assert_close(
        gradient_output["weight"],
        model.parameter_values["weight"] * scale.square(),
    )
    torch.testing.assert_close(
        hvp_output["weight"],
        vector["weight"] * scale.square(),
    )


def test_public_bind_rejects_product_without_batch_vector_inputs() -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    dense_metric = vp.metric.dense(
        matrix=torch.eye(4, dtype=torch.float64),
    )

    with pytest.raises(vp.MaterializationError, match="bind requires"):
        vp.gradient(model, loss).bind(batch={})

    with pytest.raises(vp.MaterializationError, match="bind requires"):
        vp.metric_vp(model, dense_metric).bind(batch={})


def test_public_bound_operator_compiles_vector_step_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = []

    def compile_recorder(
        operation: Callable[..., Any],
        *,
        backend: str,
        mode: str | None,
        fullgraph: bool,
        dynamic: bool | None,
        options: Mapping[str, object] | None,
    ) -> Callable[..., Any]:
        assert backend == "inductor"
        assert mode == "default"
        assert fullgraph is False
        assert dynamic is None
        assert options is None
        events.append(("compile",))

        def compiled(*args: Any, **kwargs: Any) -> Any:
            events.append(("call", len(args)))

            return operation(*args, **kwargs)

        return compiled

    monkeypatch.setattr(runtime_module.torch, "compile", compile_recorder)
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    batch = {
        "x": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        "symmetry_vector": typed_left_vector(),
    }
    vector = typed_vector()
    other_vector = {"weight": -0.25 * vector["weight"]}
    product = vp.hvp(model, loss, name="bound_hvp")
    bound = product.bind(batch=batch)
    space = vp.space.standard(
        compile=vp.Compile(
            enabled=(True,),
            boundaries=("bound_operator_vector_step",),
        )
    )

    tuned = bound.tune(
        vectors=(vector,),
        target=public_cpu_target(),
        space=space,
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    events.clear()
    first = tuned(vector)
    second = tuned(other_vector)

    assert bound.call_inputs == ("vector",)
    assert tuned.call_inputs == ("vector",)
    assert tuned.plan is not None
    bound_signature = tuned.plan.input_signature["data"]["bound_operator"]
    assert bound_signature["is_bound"] is True
    assert bound_signature["batch"]["x"] == {
        "shape": (1, 2),
        "dtype": "float64",
        "device": "cpu",
        "requires_grad": False,
    }
    assert bound_signature["batch"]["symmetry_vector"]["weight"]["shape"] == (2, 2)
    assert events == [
        ("compile",),
        ("call", 1),
        ("call", 1),
        ("call", 1),
    ]
    torch.testing.assert_close(first["weight"], 2.0 * vector["weight"])
    torch.testing.assert_close(second["weight"], 2.0 * other_vector["weight"])

    loaded = product.bind(batch=batch).load(tmp_path)
    events.clear()
    loaded_output = loaded(vector)
    stale_batch = {
        "x": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        ),
        "symmetry_vector": typed_left_vector(),
    }

    assert loaded.call_inputs == ("vector",)
    assert events == [
        ("compile",),
        ("call", 1),
        ("call", 1),
    ]
    torch.testing.assert_close(loaded_output["weight"], 2.0 * vector["weight"])

    with pytest.raises(vp.MaterializationError, match="bound operator identity"):
        product.bind(batch=stale_batch).load(tmp_path)


def test_typed_softmax_cross_entropy_gradient_and_hvp_match_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    vector = typed_vector()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")

    gradient = vp.gradient(model, loss)
    hvp = vp.hvp(model, loss)
    gradient_output = gradient(batch)
    hvp_output = hvp(batch, vector)

    logits = batch["x"] @ model.parameter_values["weight"].T
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = torch.nn.functional.one_hot(batch["labels"], num_classes=2).to(
        dtype=torch.float64
    )
    output_error = (probabilities - one_hot) / float(batch["labels"].numel())
    tangent_logits = batch["x"] @ vector["weight"].T
    hessian_tangent = probabilities * (
        tangent_logits - (probabilities * tangent_logits).sum(dim=-1, keepdim=True)
    )
    hessian_tangent = hessian_tangent / float(batch["labels"].numel())

    torch.testing.assert_close(
        gradient_output["weight"],
        output_error.T @ batch["x"],
    )
    torch.testing.assert_close(
        hvp_output["weight"],
        hessian_tangent.T @ batch["x"],
    )


def test_public_tune_builtin_softmax_cross_entropy_gradient(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    vector = {
        "weight": torch.tensor(
            [[1.0, 0.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
    }
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    product = vp.gradient(model, loss, name="loss_gradient")
    tuned = product.tune(
        data=(batch,),
        vectors=(vector,),
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    expected = product(batch)

    assert tuned.plan is not None
    torch.testing.assert_close(tuned(batch)["weight"], expected["weight"])


def test_typed_public_operators_record_typed_object_identities() -> None:
    model = typed_metric_model()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    output = vp.output("logits")
    likelihood = vp.likelihood.gaussian(
        output="logits",
        target="target",
        noise=0.5,
    )

    assert vp.gradient(model, loss).spec.semantics["loss"] == loss.signature()
    assert vp.hvp(model, loss).spec.semantics["loss"] == loss.signature()
    assert vp.ggnvp(model, loss).spec.semantics["loss"] == loss.signature()
    assert (
        vp.per_example_gradient(model, loss).spec.semantics["loss"] == loss.signature()
    )
    assert (
        vp.empirical_fisher_vp(model, loss).spec.semantics["loss"] == loss.signature()
    )
    assert vp.jvp(model, output).spec.semantics["output"] == output.signature()
    assert vp.vjp(model, output).spec.semantics["output"] == output.signature()
    assert (
        vp.fisher_vp(model, likelihood).spec.semantics["likelihood"]
        == likelihood.signature()
    )


def test_typed_softmax_cross_entropy_ggn_matches_masked_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1, 0], dtype=torch.long),
        "mask": torch.tensor([True, False, True]),
    }
    vector = typed_vector()
    loss = vp.loss.softmax_cross_entropy(
        output="logits",
        labels="labels",
        mask="mask",
    )
    operator = vp.ggnvp(model, loss)

    output = operator(batch, vector)
    expected = ce_weight_ggn_reference(
        model,
        batch,
        vector,
        mask_key="mask",
    )

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "ggnvp"
    assert "loss_hessian" not in batch
    torch.testing.assert_close(output["weight"], expected)


def test_typed_kl_loss_gradient_hvp_and_ggn_match_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "target": torch.tensor(
            [[0.75, 0.25], [0.2, 0.8]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()
    loss = vp.loss.kl(output="logits", target="target")
    gradient = vp.gradient(model, loss)
    hvp = vp.hvp(model, loss)
    ggn = vp.ggnvp(model, loss)

    gradient_output = gradient(batch)
    hvp_output = hvp(batch, vector)
    ggn_output = ggn(batch, vector)

    logits = batch["x"] @ model.parameter_values["weight"].T
    probabilities = torch.softmax(logits, dim=-1)
    target_mass = batch["target"].sum(dim=-1, keepdim=True)
    output_error = (target_mass * probabilities - batch["target"]) / float(
        batch["x"].shape[0]
    )
    tangent_logits = batch["x"] @ vector["weight"].T
    hessian_tangent = probabilities * (
        tangent_logits - (probabilities * tangent_logits).sum(dim=-1, keepdim=True)
    )
    hessian_tangent = hessian_tangent * target_mass / float(batch["x"].shape[0])

    torch.testing.assert_close(gradient_output["weight"], output_error.T @ batch["x"])
    torch.testing.assert_close(hvp_output["weight"], hessian_tangent.T @ batch["x"])
    torch.testing.assert_close(ggn_output["weight"], hessian_tangent.T @ batch["x"])


def test_typed_mse_loss_gradient_hvp_and_ggn_match_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "target": torch.tensor(
            [[0.5, -1.0], [1.25, 2.5]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()
    loss = vp.loss.mse(output="logits", target="target")
    gradient = vp.gradient(model, loss)
    hvp = vp.hvp(model, loss)
    ggn = vp.ggnvp(model, loss)

    gradient_output = gradient(batch)
    hvp_output = hvp(batch, vector)
    ggn_output = ggn(batch, vector)

    logits = batch["x"] @ model.parameter_values["weight"].T
    output_error = 2.0 * (logits - batch["target"]) / float(logits.numel())
    tangent_logits = batch["x"] @ vector["weight"].T
    hessian_tangent = 2.0 * tangent_logits / float(logits.numel())

    torch.testing.assert_close(gradient_output["weight"], output_error.T @ batch["x"])
    torch.testing.assert_close(hvp_output["weight"], hessian_tangent.T @ batch["x"])
    torch.testing.assert_close(ggn_output["weight"], hessian_tangent.T @ batch["x"])


def test_typed_declared_psd_loss_ggn_matches_dense_factor_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()
    factor = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, -1.0],
            [2.0, 0.25],
        ],
        dtype=torch.float64,
    )
    loss = vp.loss.declared_psd(output="logits", factors=factor)
    operator = vp.ggnvp(model, loss)

    output = operator(batch, vector)

    tangent_logits = batch["x"] @ vector["weight"].T
    loss_hessian = factor @ factor.T
    output_cotangent = (loss_hessian @ tangent_logits.reshape(-1)).reshape_as(
        tangent_logits
    )
    expected = output_cotangent.T @ batch["x"]

    torch.testing.assert_close(output["weight"], expected)


def test_typed_declared_psd_matrix_free_loss_ggn_matches_matvec_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()
    loss = vp.loss.declared_psd_matrix_free(
        output="logits",
        matvec=declared_psd_matrix_free_output_matvec,
        version="v1",
    )
    operator = vp.ggnvp(model, loss)

    output = operator(batch, vector)

    diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    tangent_logits = batch["x"] @ vector["weight"].T
    output_cotangent = (diagonal * tangent_logits.reshape(-1)).reshape_as(
        tangent_logits
    )
    expected = output_cotangent.T @ batch["x"]

    assert loss.signature()["identity"]["version"] == "v1"
    assert (
        "declared_psd_matrix_free_output_matvec"
        in loss.signature()["identity"]["matvec"]
    )
    torch.testing.assert_close(output["weight"], expected)


def test_typed_declared_psd_matrix_free_rejects_indefinite_matvec() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()

    def indefinite_matvec(output: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        _ = output
        diagonal = torch.tensor(
            [1.0, -1.0, 1.0, 1.0],
            dtype=tangent.dtype,
            device=tangent.device,
        )

        return (diagonal * tangent.reshape(-1)).reshape_as(tangent)

    loss = vp.loss.declared_psd_matrix_free(
        output="logits",
        matvec=indefinite_matvec,
        version="v1",
    )

    with pytest.raises(vp.MaterializationError, match="PSD output-space metric"):
        vp.ggnvp(model, loss)(batch, vector)


def test_declared_psd_matrix_free_rejects_orthogonal_indefinite_matvec() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()

    def indefinite_matvec(output: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        _ = output
        matrix = torch.eye(tangent.numel(), dtype=tangent.dtype, device=tangent.device)
        matrix[0, 0] = 0.0
        matrix[0, 1] = 1.0
        matrix[1, 0] = 1.0
        matrix[1, 1] = 0.0

        return (matrix @ tangent.reshape(-1)).reshape_as(tangent)

    loss = vp.loss.declared_psd_matrix_free(
        output="logits",
        matvec=indefinite_matvec,
        version="v1",
    )

    with pytest.raises(vp.MaterializationError, match="PSD output-space metric"):
        vp.ggnvp(model, loss)(batch, vector)


def test_declared_psd_matrix_free_certificate_uses_lanczos_residual_bound() -> None:
    matrix = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float64))
    smallest_ritz, residual_norm = public_module._lanczos_smallest_ritz_bound(
        matrix,
        matrix.shape[0],
    )

    assert smallest_ritz - residual_norm < 0.0


def test_public_matrix_free_metric_tunes_selected_curvature_product(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1, 0], dtype=torch.long),
        "symmetry_vector": typed_left_vector(),
    }
    vector = typed_vector()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    curvature = vp.ggnvp(model, loss, name="curvature")
    metric = vp.metric.matrix_free(operator=curvature)
    metric_product = vp.metric_vp(model, metric, name="curvature_metric")
    metric_inner = vp.metric_inner_vp(
        model,
        metric,
        name="curvature_metric_inner",
    )
    damping = 0.5
    inverse_product = vp.inverse_metric_vp(
        model,
        metric,
        name="curvature_inverse",
        damping=vp.damping.scalar(damping),
    )
    inverse_inner = vp.inverse_metric_inner_vp(
        model,
        metric,
        name="curvature_inverse_inner",
        damping=vp.damping.scalar(damping),
    )

    with pytest.raises(vp.MaterializationError, match="selected sibling rows"):
        metric_product(batch, vector)

    with pytest.raises(vp.MaterializationError, match="positive damping"):
        vp.inverse_metric_vp(
            model,
            metric,
            name="zero_damped_inverse",
            damping=vp.damping.scalar(0.0),
        )

    run = vp.tune(
        products=(
            curvature,
            metric_product,
            metric_inner,
            inverse_product,
            inverse_inner,
        ),
        model=model,
        data={
            "curvature": (batch,),
            "curvature_metric": (batch,),
            "curvature_metric_inner": (batch,),
            "curvature_inverse": (batch,),
            "curvature_inverse_inner": (batch,),
        },
        vectors={
            "curvature": (vector,),
            "curvature_metric": (vector,),
            "curvature_metric_inner": ((typed_left_vector(), vector),),
            "curvature_inverse": (vector,),
            "curvature_inverse_inner": ((typed_left_vector(), vector),),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    metric_output = run["curvature_metric"](batch, vector)
    metric_inner_output = run["curvature_metric_inner"](
        batch,
        typed_left_vector(),
        vector,
    )
    inverse_output = run["curvature_inverse"](batch, vector)
    inverse_inner_output = run["curvature_inverse_inner"](
        batch,
        typed_left_vector(),
        vector,
    )
    dense_matrix = dense_weight_operator_matrix(curvature, batch, vector)
    damped = dense_matrix + damping * torch.eye(
        dense_matrix.shape[0],
        dtype=dense_matrix.dtype,
    )
    flat_left = flat_weight(typed_left_vector())
    expected_inverse = torch.linalg.solve(damped, flat_weight(vector))

    assert metric_product.call_inputs == ("batch", "vector")
    assert metric_inner.call_inputs == ("batch", "left", "right")
    assert inverse_product.call_inputs == ("batch", "vector")
    assert inverse_inner.call_inputs == ("batch", "left", "right")
    assert run.plan.dependencies_by_family["curvature_metric"] == ("curvature",)
    assert run.plan.dependencies_by_family["curvature_metric_inner"] == ("curvature",)
    assert run.plan.dependencies_by_family["curvature_inverse"] == ("curvature",)
    assert run.plan.dependencies_by_family["curvature_inverse_inner"] == ("curvature",)
    assert "matrix_free_bindings" in run.plan.runtime_identities["curvature_metric"]
    torch.testing.assert_close(
        flat_weight(metric_output), flat_weight(curvature(batch, vector))
    )
    torch.testing.assert_close(
        metric_inner_output,
        flat_left @ flat_weight(metric_output),
    )
    torch.testing.assert_close(flat_weight(inverse_output), expected_inverse)
    torch.testing.assert_close(inverse_inner_output, flat_left @ expected_inverse)


def test_public_matrix_free_metric_square_roots_tune_selected_curvature_product(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    residual = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    batch = {
        "x": x,
        "target": (x @ model.parameter_values["weight"].T).detach() + residual,
    }
    vector = typed_vector()
    damping = 0.5
    likelihood = vp.likelihood.gaussian(output="logits", target="target", noise=1.0)
    curvature = vp.fisher_vp(model, likelihood, name="curvature")
    metric = vp.metric.matrix_free(operator=curvature)
    sqrt_product = vp.sqrt_metric_vp(model, metric, name="curvature_sqrt")
    inverse_sqrt_product = vp.inverse_sqrt_metric_vp(
        model,
        metric,
        name="curvature_inverse_sqrt",
        damping=vp.damping.scalar(damping),
    )

    run = vp.tune(
        products=(curvature, sqrt_product, inverse_sqrt_product),
        model=model,
        data={
            "curvature": (batch,),
            "curvature_sqrt": (batch,),
            "curvature_inverse_sqrt": (batch,),
        },
        vectors={
            "curvature": (vector,),
            "curvature_sqrt": (vector,),
            "curvature_inverse_sqrt": (vector,),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    sqrt_output = run["curvature_sqrt"](batch, vector)
    inverse_sqrt_output = run["curvature_inverse_sqrt"](batch, vector)
    dense_matrix = dense_weight_operator_matrix(curvature, batch, vector)
    damped = dense_matrix + damping * torch.eye(
        dense_matrix.shape[0],
        dtype=dense_matrix.dtype,
    )
    expected_sqrt = matrix_square_root_product(dense_matrix, flat_weight(vector))
    expected_inverse_sqrt = matrix_square_root_product(
        damped,
        flat_weight(vector),
        inverse=True,
    )

    assert sqrt_product.call_inputs == ("batch", "vector")
    assert inverse_sqrt_product.call_inputs == ("batch", "vector")
    assert run.plan.dependencies_by_family["curvature_sqrt"] == ("curvature",)
    assert run.plan.dependencies_by_family["curvature_inverse_sqrt"] == ("curvature",)
    torch.testing.assert_close(flat_weight(sqrt_output), expected_sqrt)
    torch.testing.assert_close(flat_weight(inverse_sqrt_output), expected_inverse_sqrt)


def test_typed_softmax_cross_entropy_per_example_gradient_matches_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    operator = vp.per_example_gradient(model, loss)

    output = operator(batch)
    expected_rows = ce_weight_gradient_rows(model, batch)

    assert operator.call_inputs == ("batch",)
    assert operator.spec.kind == "per_example_gradient"
    torch.testing.assert_close(output["weight"], expected_rows)


def test_typed_softmax_cross_entropy_empirical_fisher_matches_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    vector = typed_vector()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
    operator = vp.empirical_fisher_vp(model, loss)

    output = operator(batch, vector)
    rows = ce_weight_gradient_rows(model, batch).reshape(batch["x"].shape[0], -1)
    flat_vector = vector["weight"].reshape(-1)
    expected = rows.T @ (rows @ flat_vector) / float(batch["x"].shape[0])

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "empirical_fisher_vp"
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_typed_gaussian_fisher_matches_score_gradient_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "target": torch.tensor(
            [[0.25, -1.0], [1.5, 0.5], [-0.25, 0.75]],
            dtype=torch.float64,
        ),
    }
    vector = typed_vector()
    noise = 2.0
    likelihood = vp.likelihood.gaussian(
        output="logits",
        target="target",
        noise=noise,
    )
    operator = vp.fisher_vp(model, likelihood)

    output = operator(batch, vector)
    prediction = batch["x"] @ model.parameter_values["weight"].T
    residual = (batch["target"] - prediction) / (noise * noise)
    score_rows = torch.stack(
        tuple(
            torch.outer(residual[index], batch["x"][index])
            for index in range(batch["x"].shape[0])
        )
    ).reshape(batch["x"].shape[0], -1)
    flat_vector = vector["weight"].reshape(-1)
    expected = score_rows.T @ (score_rows @ flat_vector) / float(batch["x"].shape[0])

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "fisher_vp"
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_typed_gaussian_sampled_fisher_table_matches_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "target": torch.zeros((3, 2), dtype=torch.float64),
    }
    sample_table = torch.tensor(
        [
            [[0.5, -1.0], [1.25, 0.25]],
            [[1.5, 0.5], [-0.5, 1.0]],
            [[-0.25, 0.75], [0.0, -1.25]],
        ],
        dtype=torch.float64,
    )
    vector = typed_vector()
    noise = 2.0
    likelihood = vp.likelihood.gaussian(
        output="logits",
        target="target",
        noise=noise,
    )
    source = vp.samples.table(table=sample_table, identity={"table": "gaussian-v1"})
    operator = vp.sampled_fisher_vp(model, likelihood, samples=source)

    output = operator(batch, vector)
    prediction = batch["x"] @ model.parameter_values["weight"].T
    rows = []

    for example_index in range(batch["x"].shape[0]):
        for sample_index in range(sample_table.shape[1]):
            residual = (
                sample_table[example_index, sample_index] - prediction[example_index]
            ) / (noise * noise)
            rows.append(torch.outer(residual, batch["x"][example_index]).reshape(-1))

    score_rows = torch.stack(tuple(rows))
    flat_vector = vector["weight"].reshape(-1)
    expected = score_rows.T @ (score_rows @ flat_vector) / float(score_rows.shape[0])

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "sampled_fisher_vp"
    assert operator.spec.semantics["sample_source_identity"]["identity"][
        "identity"
    ] == {"table": "gaussian-v1"}
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_typed_categorical_sampled_fisher_table_matches_reference() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    sample_table = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    vector = typed_vector()
    likelihood = vp.likelihood.categorical(output="logits", labels="labels")
    source = vp.samples.table(table=sample_table, identity="categorical-v1")
    operator = vp.sampled_fisher_vp(model, likelihood, samples=source)

    output = operator(batch, vector)
    logits = batch["x"] @ model.parameter_values["weight"].T
    probabilities = torch.softmax(logits, dim=-1)
    rows = []

    for example_index in range(batch["x"].shape[0]):
        for sample_index in range(sample_table.shape[1]):
            label = int(sample_table[example_index, sample_index].item())
            score = -probabilities[example_index].clone()
            score[label] = score[label] + 1.0
            rows.append(torch.outer(score, batch["x"][example_index]).reshape(-1))

    score_rows = torch.stack(tuple(rows))
    flat_vector = vector["weight"].reshape(-1)
    expected = score_rows.T @ (score_rows @ flat_vector) / float(score_rows.shape[0])

    assert operator.call_inputs == ("batch", "vector")
    assert operator.spec.kind == "sampled_fisher_vp"
    assert operator.spec.semantics["denominator"] == "num_tokens"
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_typed_sampled_fisher_fixed_seed_repeats() -> None:
    model = typed_metric_model()
    batch = {
        "x": torch.tensor(
            [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
            dtype=torch.float64,
        ),
        "target": torch.zeros((3, 2), dtype=torch.float64),
    }
    vector = typed_vector()
    likelihood = vp.likelihood.gaussian(
        output="logits",
        target="target",
        noise=1.5,
    )
    source = vp.samples.fixed_seed(seed=17, count=3)
    other_source = vp.samples.fixed_seed(seed=19, count=3)
    operator = vp.sampled_fisher_vp(model, likelihood, samples=source)
    other_operator = vp.sampled_fisher_vp(model, likelihood, samples=other_source)

    first = operator(batch, vector)
    second = operator(batch, vector)
    other = other_operator(batch, vector)

    assert operator.spec.semantics["sample_source_identity"]["identity"] == {
        "seed": 17,
        "count": 3,
    }
    assert operator.spec.semantics["sampling_bound"] == {"kind": "disabled"}
    torch.testing.assert_close(first["weight"], second["weight"])

    with pytest.raises(AssertionError):
        torch.testing.assert_close(first["weight"], other["weight"])


def test_typed_sampled_fisher_sampling_bound_enters_identity() -> None:
    model = typed_metric_model()
    likelihood = vp.likelihood.gaussian(
        output="logits",
        target="target",
        noise=1.5,
    )
    source = vp.samples.fixed_seed(
        seed=17,
        count=3,
        sampling_bound={
            "kind": "matrix_bernstein",
            "failure_probability": 0.25,
            "norm_floor": 1e-12,
        },
    )
    operator = vp.sampled_fisher_vp(model, likelihood, samples=source)

    assert operator.spec.semantics["sampling_bound"] == {
        "kind": "matrix_bernstein",
        "failure_probability": pytest.approx(0.25),
        "norm_floor": pytest.approx(1e-12),
    }
    assert (
        operator.spec.semantics["sample_source_identity"]["sampling_bound"]
        == operator.spec.semantics["sampling_bound"]
    )


def test_typed_sample_source_validation() -> None:
    with pytest.raises(vp.MaterializationError, match="seed"):
        vp.samples.fixed_seed(seed=True, count=2)

    with pytest.raises(vp.MaterializationError, match="count"):
        vp.samples.fixed_seed(seed=1, count=0)

    with pytest.raises(vp.MaterializationError, match="example and sample axes"):
        vp.samples.table(table=torch.ones(2), identity="bad")

    with pytest.raises(vp.MaterializationError, match="JSON-compatible"):
        vp.samples.table(table=torch.ones(2, 1), identity=object())

    with pytest.raises(vp.MaterializationError, match="failure_probability"):
        vp.samples.fixed_seed(
            seed=1,
            count=2,
            sampling_bound={
                "kind": "hutchinson_relative_variance",
                "failure_probability": 1.0,
                "norm_floor": 1e-12,
            },
        )


def test_typed_categorical_fisher_routes_to_ggn() -> None:
    model = typed_metric_model()
    likelihood = vp.likelihood.categorical(output="logits", labels="labels")

    with pytest.raises(vp.MaterializationError, match="GGNVP"):
        vp.fisher_vp(model, likelihood)


def test_typed_per_example_gradient_rejects_scalar_loss_without_rows() -> None:
    model = typed_metric_model()

    def scalar_loss(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        _ = buffers, batch, context

        return params["weight"].square().sum()

    loss = vp.loss.from_scalar(scalar_loss, output="logits", version="scalar-v1")

    with pytest.raises(vp.MaterializationError, match="per-example lowering"):
        vp.per_example_gradient(model, loss)


def test_typed_empirical_fisher_rejects_scalar_loss_without_rows() -> None:
    model = typed_metric_model()

    def scalar_loss(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        _ = buffers, batch, context

        return params["weight"].square().sum()

    loss = vp.loss.from_scalar(scalar_loss, output="logits", version="scalar-v1")

    with pytest.raises(vp.MaterializationError, match="per-example lowering"):
        vp.empirical_fisher_vp(model, loss)


def test_typed_ggn_rejects_scalar_loss_without_output_hessian() -> None:
    model = typed_metric_model()

    def scalar_loss(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        _ = buffers, batch, context

        return params["weight"].square().sum()

    loss = vp.loss.from_scalar(scalar_loss, output="logits", version="scalar-v1")

    with pytest.raises(vp.MaterializationError, match="output-Hessian lowering"):
        vp.ggnvp(model, loss)


def test_typed_softmax_cross_entropy_rejects_invalid_fields() -> None:
    with pytest.raises(vp.MaterializationError, match="loss output"):
        vp.loss.softmax_cross_entropy(output="", labels="labels")

    with pytest.raises(vp.MaterializationError, match="loss reduction"):
        vp.loss.softmax_cross_entropy(
            output="logits",
            labels="labels",
            reduction="average_tokens",
        )

    with pytest.raises(vp.MaterializationError, match="denominator"):
        vp.loss.softmax_cross_entropy(
            output="logits",
            labels="labels",
            denominator="examples",
        )

    with pytest.raises(vp.MaterializationError, match="loss target"):
        vp.loss.kl(output="logits", target="")

    with pytest.raises(vp.MaterializationError, match="denominator"):
        vp.loss.kl(output="logits", target="target", denominator="examples")

    with pytest.raises(vp.MaterializationError, match="denominator"):
        vp.loss.mse(output="logits", target="target", denominator="num_tokens")

    with pytest.raises(vp.MaterializationError, match="factors"):
        vp.loss.declared_psd(output="logits", factors=torch.ones(2))

    with pytest.raises(vp.MaterializationError, match="matvec"):
        vp.loss.declared_psd_matrix_free(
            output="logits",
            matvec=non_callable_value(),
            version="v1",
        )

    with pytest.raises(vp.MaterializationError, match="loss version"):
        vp.loss.declared_psd_matrix_free(
            output="logits",
            matvec=declared_psd_matrix_free_output_matvec,
            version="",
        )


def test_typed_case_copies_batch_and_validates_vector() -> None:
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    probe = vp.case(batch=batch, vector=vector)
    batch["extra"] = torch.tensor([1.0], dtype=torch.float64)

    assert probe.batch is not None
    assert set(probe.batch) == {"x"}
    torch.testing.assert_close(
        probe.batch["x"],
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )
    assert probe.vector is vector


def test_public_search_builders_lower_to_engine_policies() -> None:
    admission = vp.search.admission()
    balanced = vp.search.balanced(retain=3, compile_horizons=(1, 10))
    thorough = vp.search.thorough(
        retain=2,
        compile_horizons=(1, 5),
        variance_repeats=2,
    )

    assert admission.policy.strategy == "admission"
    assert balanced.policy.strategy == "balanced"
    assert balanced.policy.retained_top_count == 3
    assert balanced.policy.compile_call_horizons == (1, 10)
    assert thorough.policy.strategy == "thorough"
    assert thorough.policy.retained_top_count == 2
    assert thorough.policy.compile_call_horizons == (1, 5)
    assert thorough.policy.variance_repeat_count == 2

    with pytest.raises(vp.MaterializationError, match="retained_top_count"):
        vp.search.balanced(retain=0)

    with pytest.raises(vp.MaterializationError, match="variance_repeat_count"):
        vp.search.thorough(
            retain=2,
            compile_horizons=(1,),
            variance_repeats=1,
        )


def public_target_with_overrides(**overrides: Any) -> vp.Target:
    return dataclasses.replace(public_cpu_target(), **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"devices": ()}, "target devices"),
        ({"devices": ("cpu", "cpu")}, "target devices"),
        ({"accelerator": ""}, "accelerator"),
        ({"allowed_dtypes": ("fp32", "fp32")}, "allowed_dtypes"),
        (
            {"allowed_sdpa_kernels": unchecked_public_value(["math"])},
            "allowed_sdpa_kernels",
        ),
        ({"allowed_sharding_modes": ("",)}, "allowed_sharding_modes"),
    ],
)
def test_public_target_rejects_invalid_identity_fields(
    overrides: Mapping[str, Any],
    message: str,
) -> None:
    with pytest.raises(vp.MaterializationError, match=message):
        public_target_with_overrides(**overrides)


def test_public_policy_identity_fields_must_be_json_compatible() -> None:
    with pytest.raises(vp.MaterializationError, match="determinism policy key"):
        vp.DeterminismPolicy(unchecked_public_value({1: "enabled"}))

    with pytest.raises(vp.MaterializationError, match="environment policy fields"):
        vp.EnvironmentPolicy({"bad": object()})


def test_public_cuda_target_lowers_to_engine_target() -> None:
    target = vp.cuda(
        0,
        "h100",
        timing=vp.TimingPolicy(short_measured_calls=1),
        selection=vp.SelectionPolicy(compile_call_horizon=3),
        determinism=vp.DeterminismPolicy({"tf32": False}),
        environment=vp.EnvironmentPolicy({"CUDA_VISIBLE_DEVICES": "0"}),
    )
    lower = target.lower(vp.search.fast())

    assert lower.devices == ("cuda:0",)
    assert lower.accelerator == "h100"
    assert lower.timing_policy.short_measured_calls == 1
    assert lower.selection_policy.compile_call_horizon == 3
    assert lower.search_policy.strategy == "fast"
    assert lower.determinism_policy == {"tf32": False}
    assert lower.environment_capture == {"CUDA_VISIBLE_DEVICES": "0"}
    assert "fp32" in lower.allowed_dtypes
    assert "fp8_when_supported" in lower.allowed_dtypes
    assert "pytorch_sdpa_direct" in lower.allowed_attention_frontends
    assert "math" in lower.allowed_sdpa_kernels
    assert "fsdp2" in lower.allowed_sharding_modes


def test_public_problem_autotune_and_operator_load_replay(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    product = vp.gradient(model, loss, name="grad_product")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    target = public_cpu_target()
    search = vp.search.exhaustive()
    space = vp.space.standard()
    run_dir = tmp_path / "operator"
    vector = typed_vector()

    tuning_problem = vp.problem(
        product,
        data=(batch,),
        vectors=(vector,),
        target=target,
        space=space,
        search=search,
    )
    problem_plan = vp.autotune(tuning_problem, run_dir=tmp_path / "problem")
    tuned = product.tune(
        data=(batch,),
        vectors=(vector,),
        target=target,
        space=space,
        search=search,
        run_dir=run_dir,
    )
    loaded = vp.gradient(model, loss, name="grad_product").load(run_dir)

    expected = {"weight": 2.0 * model.parameter_values["weight"]}
    problem_selected = problem_plan.materialize(name="grad_product")

    torch.testing.assert_close(
        problem_selected(batch, {})["weight"], expected["weight"]
    )
    torch.testing.assert_close(tuned(batch)["weight"], expected["weight"])
    torch.testing.assert_close(loaded(batch)["weight"], expected["weight"])


def test_public_operator_load_rejects_stale_model_identity(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    product = vp.gradient(model, loss, name="grad_product")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    run_dir = tmp_path / "operator"
    vector = typed_vector()

    product.tune(
        data=(batch,),
        vectors=(vector,),
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=run_dir,
    )
    model.module.eval()

    with pytest.raises(vp.MaterializationError, match="model identity differs"):
        vp.gradient(model, loss, name="grad_product").load(run_dir)


def test_public_tune_returns_run_with_tuned_product(tmp_path: Path) -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    product = vp.gradient(model, loss, name="grad_product")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    run = vp.tune(
        products=(product,),
        model=model,
        data={"grad_product": (batch,)},
        vectors={"grad_product": (vector,)},
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    expected = {"weight": 2.0 * model.parameter_values["weight"]}

    assert "grad_product" in run
    assert len(run) == 1
    torch.testing.assert_close(
        run.plan.materialize(name="grad_product")(batch, {})["weight"],
        expected["weight"],
    )
    torch.testing.assert_close(run["grad_product"](batch)["weight"], expected["weight"])


def test_public_tune_accepts_case_reference_and_probe_inputs(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    scale = torch.full_like(model.parameter_values["weight"], 2.0)
    probe_scale = torch.full_like(model.parameter_values["weight"], 3.0)
    data_batch = {"scale": torch.full_like(model.parameter_values["weight"], 4.0)}
    reference_batch = {"scale": scale, "symmetry_vector": typed_left_vector()}
    probe_batch = {"scale": probe_scale}
    vector = typed_vector()

    def quadratic(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        _ = buffers, context

        return 0.5 * (params["weight"] * batch["scale"]).square().sum()

    loss = vp.loss.from_scalar(quadratic, output="logits", version="quadratic-v1")
    product = vp.hvp(model, loss, name="hvp_product")
    run = vp.tune(
        products=(product,),
        model=model,
        data=(data_batch,),
        vectors=(vector,),
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
        reference=vp.case(batch=reference_batch, vector=vector),
        probes=(vp.case(batch=probe_batch, vector=vector),),
    )
    data_signature = run.plan.input_signature["data"]
    vector_signature = run.plan.input_signature["vectors"]
    result = run["hvp_product"](reference_batch, vector)

    assert data_signature["reference"]["scale"]["shape"] == (2, 2)
    assert data_signature["probe_batches"][0]["scale"]["shape"] == (2, 2)
    assert vector_signature["probe_vectors"][0]["weight"]["shape"] == (2, 2)
    torch.testing.assert_close(result["weight"], vector["weight"] * scale.square())


def test_public_tune_returns_run_with_two_products(tmp_path: Path) -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    first = vp.gradient(model, loss, name="first_grad")
    second = vp.gradient(model, loss, name="second_grad")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    run = vp.tune(
        products=(first, second),
        model=model,
        data={"first_grad": (batch,), "second_grad": (batch,)},
        vectors={"first_grad": (vector,), "second_grad": (vector,)},
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    expected = {"weight": 2.0 * model.parameter_values["weight"]}

    assert tuple(run) == ("first_grad", "second_grad")
    torch.testing.assert_close(run["first_grad"](batch)["weight"], expected["weight"])
    torch.testing.assert_close(run["second_grad"](batch)["weight"], expected["weight"])


def test_public_tune_guards_root_product_set_arguments() -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    product = vp.gradient(model, loss, name="grad_product")
    duplicate = vp.gradient(model, loss, name="grad_product")
    mismatched_model = vp.torch_model(
        model.module,
        parameters=model.parameters,
        call=vp.module_call(args=("x",), kwargs={}, output="other_logits"),
    )
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    with pytest.raises(vp.MaterializationError, match="at least one product"):
        vp.tune(
            products=(),
            model=model,
            data=(),
            vectors=(),
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )

    with pytest.raises(vp.MaterializationError, match="names must be unique"):
        vp.tune(
            products=(product, duplicate),
            model=model,
            data={"grad_product": (batch,)},
            vectors={"grad_product": (vector,)},
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )

    with pytest.raises(vp.MaterializationError, match="product model differs"):
        vp.tune(
            products=(product,),
            model=mismatched_model,
            data=(batch,),
            vectors=(vector,),
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )


def test_public_tune_guards_root_product_data_and_vector_arguments() -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    first = vp.gradient(model, loss, name="first_grad")
    second = vp.gradient(model, loss, name="second_grad")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    with pytest.raises(vp.MaterializationError, match="data is missing product"):
        vp.tune(
            products=(first, second),
            model=model,
            data={"first_grad": (batch,)},
            vectors={"first_grad": (vector,), "second_grad": (vector,)},
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )

    with pytest.raises(vp.MaterializationError, match="vectors are missing product"):
        vp.tune(
            products=(first, second),
            model=model,
            data={"first_grad": (batch,), "second_grad": (batch,)},
            vectors={"first_grad": (vector,)},
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )

    with pytest.raises(vp.MaterializationError, match="vectors must be keyed"):
        vp.tune(
            products=(first, second),
            model=model,
            data={"first_grad": (batch,), "second_grad": (batch,)},
            vectors=(vector,),
            target=public_cpu_target(),
            space=vp.space.standard(),
            search=vp.search.exhaustive(),
        )


def test_public_tune_multi_product_cohort_without_run_dir() -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    first = vp.gradient(model, loss, name="first_grad")
    second = vp.gradient(model, loss, name="second_grad")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()
    space = vp.SearchSpace(axes={"layout.params": ("parameter_tree",)})
    run = vp.tune(
        products=(first, second),
        model=model,
        data={
            "first_grad": (batch,),
            "second_grad": (batch,),
        },
        vectors={
            "first_grad": (vector,),
            "second_grad": (vector,),
        },
        target=public_cpu_target(),
        space=space,
        search=vp.search.exhaustive(),
        cohort_constraints=(vp.cohort.layout_coherence(("layout.params",)),),
        run_dir=None,
        reference=vp.case(batch=batch, vector=vector),
        probes=(vp.case(batch=batch, vector=vector),),
    )
    expected = {"weight": 2.0 * model.parameter_values["weight"]}

    assert run.plan.run_dir is None
    assert run.plan.cohort_assignment is not None
    assert set(run.plan.selected) == {"first_grad", "second_grad"}
    assert run.plan.input_signature["first_grad"]["data"]["reference"]["x"][
        "shape"
    ] == (1, 2)
    assert run.plan.input_signature["second_grad"]["vectors"]["probe_vectors"][0][
        "weight"
    ]["shape"] == (2, 2)
    torch.testing.assert_close(run["first_grad"](batch)["weight"], expected["weight"])
    torch.testing.assert_close(run["second_grad"](batch)["weight"], expected["weight"])


def test_public_tune_composition_uses_selected_child_rows(tmp_path: Path) -> None:
    model = typed_metric_model()
    curvature_matrix = torch.diag(
        torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    )
    preconditioner_matrix = torch.diag(
        torch.tensor([11.0, 13.0, 17.0, 19.0], dtype=torch.float64)
    )
    curvature = vp.metric_vp(
        model,
        vp.metric.dense(matrix=curvature_matrix),
        name="curvature",
    )
    preconditioner = vp.metric_vp(
        model,
        vp.metric.dense(matrix=preconditioner_matrix),
        name="preconditioner",
    )
    composed = vp.composition(
        model,
        name="preconditioned_curvature",
        children=("curvature", "preconditioner"),
        combine=vp.compose("preconditioner", "curvature"),
    )
    vector = typed_vector()
    batch = {}

    run = vp.tune(
        products=(curvature, preconditioner, composed),
        model=model,
        data={
            "curvature": (batch,),
            "preconditioner": (batch,),
            "preconditioned_curvature": (batch,),
        },
        vectors={
            "curvature": (vector,),
            "preconditioner": (vector,),
            "preconditioned_curvature": (vector,),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    output = run["preconditioned_curvature"](batch, vector)
    expected = preconditioner_matrix @ (curvature_matrix @ flat_weight(vector))

    assert run.plan.dependencies_by_family["preconditioned_curvature"] == (
        "curvature",
        "preconditioner",
    )
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_public_tune_linear_combination_composition_executes_scaled_identity(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    curvature_matrix = torch.diag(
        torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    )
    curvature = vp.metric_vp(
        model,
        vp.metric.dense(matrix=curvature_matrix),
        name="curvature",
    )
    damped = vp.composition(
        model,
        name="damped_curvature",
        children=("curvature",),
        combine=vp.linear_combination(
            (1.5, "curvature"),
            (-0.25, vp.scaled_identity(2.0)),
        ),
    )
    vector = typed_vector()
    batch = {}

    run = vp.tune(
        products=(curvature, damped),
        model=model,
        data={
            "curvature": (batch,),
            "damped_curvature": (batch,),
        },
        vectors={
            "curvature": (vector,),
            "damped_curvature": (vector,),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    output = run["damped_curvature"](batch, vector)
    replay_context = public_replay_context_for_plan(run.plan)
    replayed = vp.load_plan(
        tmp_path,
        replay_context=replay_context,
        materializers=run.plan.materializers,
    )
    replayed_output = replayed.materialize(name="damped_curvature")(batch, vector)
    expected = 1.5 * (curvature_matrix @ flat_weight(vector)) - 0.5 * flat_weight(
        vector
    )
    stale_context = replay_context_with_stale_composition_coefficient(
        replay_context,
        "damped_curvature",
        1.25,
    )

    assert run.plan.dependencies_by_family["damped_curvature"] == ("curvature",)
    torch.testing.assert_close(output["weight"].reshape(-1), expected)
    torch.testing.assert_close(replayed_output["weight"].reshape(-1), expected)

    with pytest.raises(vp.StaleRecordError):
        vp.load_plan(
            tmp_path,
            replay_context=stale_context,
            materializers=run.plan.materializers,
        )


def test_public_tune_source_composition_executes_batch_to_vector_child(
    tmp_path: Path,
) -> None:
    model = typed_metric_model()
    loss = vp.loss.from_scalar(
        squared_weight_loss,
        output="logits",
        version="squared-weight-v1",
    )
    curvature_matrix = torch.diag(
        torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    )
    gradient = vp.gradient(model, loss, name="gradient")
    curvature = vp.metric_vp(
        model,
        vp.metric.dense(matrix=curvature_matrix),
        name="curvature",
    )
    curvature_gradient = vp.composition(
        model,
        name="curvature_gradient",
        children=("gradient", "curvature"),
        combine=vp.compose("curvature", vp.source("gradient")),
    )
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()

    run = vp.tune(
        products=(gradient, curvature, curvature_gradient),
        model=model,
        data={
            "gradient": (batch,),
            "curvature": ({},),
            "curvature_gradient": (batch,),
        },
        vectors={
            "gradient": (vector,),
            "curvature": (vector,),
            "curvature_gradient": ({},),
        },
        target=public_cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    output = run["curvature_gradient"](batch)
    gradient_vector = 2.0 * flat_weight(model.parameter_values)
    expected = curvature_matrix @ gradient_vector

    assert curvature_gradient.call_inputs == ("batch",)
    assert run.plan.dependencies_by_family["curvature_gradient"] == (
        "gradient",
        "curvature",
    )
    torch.testing.assert_close(output["weight"].reshape(-1), expected)


def test_public_cohort_layout_coherence_lowers_from_search_space() -> None:
    rule = vp.cohort.layout_coherence(("dtype.vector", "layout.vector"))
    space = vp.SearchSpace(
        axes={
            "dtype.vector": ("fp32", "bf16"),
            "layout.vector": ("flat_contiguous",),
        }
    )

    lower = rule.lower(space)

    assert lower.settings_keys == ("dtype.vector", "layout.vector")
    assert tuple(dict(assignment) for assignment in lower.assignments) == (
        {"dtype.vector": "fp32", "layout.vector": "flat_contiguous"},
        {"dtype.vector": "bf16", "layout.vector": "flat_contiguous"},
    )

    with pytest.raises(vp.MaterializationError, match="not declared"):
        rule.lower(vp.space.standard())


def test_public_space_autodiff_component_tunes_jvp_path(tmp_path: Path) -> None:
    model = typed_metric_model()
    product = vp.jvp(model, vp.output("logits"), name="jvp_product")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()
    space = vp.space.standard(autodiff=vp.AD(paths=("torch_func_jvp",)))

    tuned = product.tune(
        data=(batch,),
        vectors=(vector,),
        target=public_cpu_target(),
        space=space,
        search=vp.search.exhaustive(),
        run_dir=tmp_path,
    )
    expected = batch["x"] @ vector["weight"].T

    assert tuned.plan is not None
    assert tuned.plan.selected["jvp_product"].settings["jvp.path"] == "torch_func_jvp"
    torch.testing.assert_close(tuned(batch, vector), expected)


def test_public_space_components_generate_admitted_settings() -> None:
    model = typed_metric_model()
    product = vp.gradient(
        model,
        vp.loss.from_scalar(
            squared_weight_loss,
            output="logits",
            version="squared-weight-v1",
        ),
        name="grad_product",
    )
    space = vp.space.standard(
        autodiff=vp.AD(paths=("torch_autograd_grad",)),
        precision=vp.Precision(model=("fp32",), accumulation=("fp32",)),
        layout=vp.Layout(params=("parameter_tree",)),
        compile=vp.Compile(enabled=(False,)),
    )
    settings_by_id = space.candidate_settings(product)
    settings = next(iter(settings_by_id.values()))
    candidate = vpx.Candidate("grad_product", "public-space", settings)

    assert settings["gradient.path"] == "torch_autograd_grad"
    assert settings["dtype.model_compute"] == "fp32"
    assert settings["layout.params"] == "parameter_tree"
    assert settings["compile.enabled"] == "false"
    assert vpx.standard_axis_registry().admit(candidate).admission_status == "passed"

    with pytest.raises(vp.MaterializationError, match="manual_batch"):
        vp.Vectorization(modes=("manual_batch",))

    with pytest.raises(vp.MaterializationError, match="boundaries"):
        vp.Compile(enabled=(True,))


def test_public_space_compile_component_generates_conditional_rows() -> None:
    model = typed_metric_model()
    product = vp.gradient(
        model,
        vp.loss.from_scalar(
            squared_weight_loss,
            output="logits",
            version="squared-weight-v1",
        ),
        name="grad_product",
    )
    space = vp.space.standard(
        compile=vp.Compile(
            enabled=(False, True),
            boundaries=("gradient_closure",),
        ),
    )
    settings_by_id = space.candidate_settings(product)
    rows = tuple(settings_by_id.values())
    eager = next(row for row in rows if row["compile.enabled"] == "false")
    compiled = next(row for row in rows if row["compile.enabled"] == "true")
    registry = vpx.standard_axis_registry()

    assert len(rows) == 2
    assert "compile.boundary" not in eager
    assert compiled["compile.boundary"] == "gradient_closure"
    assert compiled["compile.backend"] == "inductor"
    assert compiled["compile.mode"] == "default"

    for row in rows:
        candidate = vpx.Candidate("grad_product", "public-compile", row)

        assert registry.admit(candidate).admission_status == "passed"


def test_public_space_with_attention_generates_admitted_adapter_settings() -> None:
    model = typed_metric_model()
    product = vp.gradient(
        model,
        vp.loss.from_scalar(
            squared_weight_loss,
            output="logits",
            version="squared-weight-v1",
        ),
        name="grad_product",
    )
    attention = vp.adapters.transformers.attention_space(
        frontends=("sdpa",),
        sdpa_kernel="math",
    )
    space = vp.space.standard(
        autodiff=vp.AD(paths=("torch_autograd_grad",)),
        precision=vp.Precision(model=("fp32",), accumulation=("fp32",)),
        layout=vp.Layout(params=("parameter_tree",)),
        compile=vp.Compile(enabled=(False,)),
    ).with_attention(attention)
    settings_by_id = space.candidate_settings(product)
    settings = next(iter(settings_by_id.values()))
    candidate = vpx.Candidate("grad_product", "public-attention", settings)
    registry = adapter_search_registry(attention.axis_descriptors())

    assert settings["attention.frontend"] == "transformers_sdpa"
    assert settings["attention.sdpa_kernel"] == "math"
    assert settings["module_mode"] == "eval"
    assert settings["dropout_p"] == pytest.approx(0.0)
    assert registry.admit(candidate).admission_status == "passed"


def test_public_space_with_distributed_generates_admitted_adapter_settings() -> None:
    model = typed_metric_model()
    product = vp.gradient(
        model,
        vp.loss.from_scalar(
            squared_weight_loss,
            output="logits",
            version="squared-weight-v1",
        ),
        name="grad_product",
    )
    distributed = vp.adapters.distributed.space(strategy=("single_gpu",))
    space = vp.space.standard(
        autodiff=vp.AD(paths=("torch_autograd_grad",)),
        precision=vp.Precision(model=("fp32",), accumulation=("fp32",)),
        layout=vp.Layout(params=("parameter_tree",)),
        compile=vp.Compile(enabled=(False,)),
    ).with_distributed(distributed)
    settings_by_id = space.candidate_settings(product)
    settings = next(iter(settings_by_id.values()))
    candidate = vpx.Candidate("grad_product", "public-distributed", settings)
    registry = adapter_search_registry(distributed.axis_descriptors())

    assert settings["distributed.strategy"] == "single_gpu"
    assert registry.admit(candidate).admission_status == "passed"


def adapter_search_registry(
    descriptors: tuple[vpx.AxisDescriptor, ...],
) -> vpx.AxisRegistry:
    descriptor_keys = set()

    for descriptor in descriptors:
        descriptor_keys.update(descriptor.settings_keys)
        descriptor_keys.update(descriptor.optional_settings_keys)

    excluded = tuple(
        axis.name
        for axis in vpx.standard_axis_descriptors()
        if descriptor_keys.intersection((
            *axis.settings_keys,
            *axis.optional_settings_keys,
        ))
    )
    registry = vpx.standard_axis_registry(exclude=excluded)

    for descriptor in descriptors:
        registry.register(descriptor)

    return registry


def test_typed_output_jvp_and_vjp_execute_reference() -> None:
    model = typed_metric_model()
    output = vp.output("logits")
    batch = {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    vector = typed_vector()
    cotangent = torch.tensor([[0.25, -0.5]], dtype=torch.float64)

    jvp = vp.jvp(model, output)
    vjp = vp.vjp(model, output)

    jvp_output = jvp(batch, vector)
    vjp_output = vjp(batch, cotangent)

    torch.testing.assert_close(
        jvp_output,
        batch["x"] @ vector["weight"].T,
    )
    torch.testing.assert_close(
        vjp_output["weight"],
        cotangent.T @ batch["x"],
    )


def test_replay_identity_fields_distinguish_declared_variants() -> None:
    model = typed_metric_model()
    loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")

    matrix_free_a = vp.metric.matrix_free(
        operator=vp.ggnvp(model, loss, name="curvature_a")
    )
    matrix_free_b = vp.metric.matrix_free(
        operator=vp.ggnvp(model, loss, name="curvature_b")
    )
    inverse_a = vp.inverse_metric_vp(
        model, matrix_free_a, damping=vp.damping.scalar(0.5)
    )
    inverse_b = vp.inverse_metric_vp(
        model, matrix_free_b, damping=vp.damping.scalar(0.5)
    )

    assert inverse_a.spec.semantics != inverse_b.spec.semantics

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: Mapping[str, object],
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        del buffers, batch, context

        return params["weight"].square().sum()

    from_scalar_v1 = vp.gradient(
        model,
        vp.loss.from_scalar(scalar_objective, output="logits", version="v1"),
    )
    from_scalar_v2 = vp.gradient(
        model,
        vp.loss.from_scalar(scalar_objective, output="logits", version="v2"),
    )

    assert from_scalar_v1.spec.semantics != from_scalar_v2.spec.semantics

    curvature = vp.ggnvp(model, loss, name="curvature")
    gradient = vp.gradient(model, loss, name="gradient")

    def weighted_composition(coefficient: float) -> vp.Operator:
        return vp.composition(
            model,
            name="weighted",
            children={"curvature": curvature, "gradient": gradient},
            combine=vp.linear_combination(
                (coefficient, "curvature"),
                (1.0, "gradient"),
            ),
        )

    assert (
        weighted_composition(1.5).spec.semantics
        != weighted_composition(2.5).spec.semantics
    )

    likelihood = vp.likelihood.gaussian(output="logits", target="target", noise=1.0)
    sampled_a = vp.sampled_fisher_vp(
        model, likelihood, samples=vp.samples.fixed_seed(seed=1, count=4)
    )
    sampled_b = vp.sampled_fisher_vp(
        model, likelihood, samples=vp.samples.fixed_seed(seed=2, count=4)
    )

    assert sampled_a.spec.semantics != sampled_b.spec.semantics

    ekfac = typed_ekfac_metric()
    inner_norm = vp.inverse_metric_inner_vp(
        model, ekfac, damping=vp.damping.scalar(0.5), as_norm=True
    )
    inner_gram = vp.inverse_metric_inner_vp(
        model, ekfac, damping=vp.damping.scalar(0.5), as_norm=False
    )

    assert inner_norm.spec.semantics != inner_gram.spec.semantics

    tol_a = vp.inverse_metric_vp(
        model, matrix_free_a, damping=vp.damping.scalar(0.5), tol=1e-4
    )
    tol_b = vp.inverse_metric_vp(
        model, matrix_free_a, damping=vp.damping.scalar(0.5), tol=2e-4
    )

    assert tol_a.spec.semantics != tol_b.spec.semantics
    assert (
        vp.inverse_metric_vp(
            model, matrix_free_a, damping=vp.damping.scalar(0.5), tol=1e-4
        ).spec.semantics
        == tol_a.spec.semantics
    )
