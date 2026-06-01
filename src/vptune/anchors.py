"""Built-in anchor registry."""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.func import functional_call, grad, jvp, vjp

from vptune.admission import admit_functional_call
from vptune.tensor_tree import (
    TensorTree,
    tree_dot,
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_map2,
    tree_mul,
    tree_sub,
)

ScalarTensorFn = Callable[[Any], torch.Tensor]
TensorFn = Callable[[Any], Any]


class Anchor(Protocol):
    """Reference implementation callable."""

    def __call__(self) -> TensorTree:
        """Return an anchor result."""


@dataclasses.dataclass(slots=True)
class AnchorRegistry:
    """Registry for package-owned anchors."""

    anchors: dict[str, Anchor] = dataclasses.field(default_factory=dict)

    def register(self, name: str, anchor: Anchor) -> None:
        """Register an anchor implementation.

        Raises:
            RuntimeError: If the anchor name is already registered.
        """
        if name in self.anchors:
            message = f"anchor already registered: {name}"
            raise RuntimeError(message)

        self.anchors[name] = anchor

    def get(self, name: str) -> Anchor:
        """Return a registered anchor.

        Returns:
            Registered anchor.

        Raises:
            RuntimeError: If the anchor name is unknown.
        """
        anchor = self.anchors.get(name)

        if anchor is None:
            message = f"anchor is not registered: {name}"
            raise RuntimeError(message)

        return anchor


def gradient_anchor(function: ScalarTensorFn, params: TensorTree) -> Any:
    """Return gradient of a scalar function."""
    active = _active_tree(params)
    value = function(active)
    leaves = tree_leaves(active)
    gradients = torch.autograd.grad(
        value,
        leaves,
        allow_unused=True,
    )

    return tree_from_leaves(
        active,
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient.detach()
            for leaf, gradient in zip(leaves, gradients, strict=True)
        ),
    )


def jvp_anchor(
    function: TensorFn,
    params: TensorTree,
    vector: TensorTree,
) -> Any:
    """Return JVP through a pure function."""
    result = jvp(function, (params,), (vector,))[1]

    return result


def vjp_anchor(
    function: TensorFn,
    params: TensorTree,
    cotangent: TensorTree,
) -> Any:
    """Return VJP through a pure function."""
    pullback = vjp(function, params)[1]
    (result,) = pullback(cotangent)

    return result


def hvp_reverse_over_reverse_anchor(
    function: ScalarTensorFn,
    params: TensorTree,
    vector: TensorTree,
) -> Any:
    """Return HVP by reverse-over-reverse AD."""
    active = _active_tree(params)
    value = function(active)
    leaves = tree_leaves(active)
    gradient_leaves = torch.autograd.grad(
        value,
        leaves,
        create_graph=True,
        allow_unused=True,
    )
    gradient_tree = tree_from_leaves(
        active,
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient
            for leaf, gradient in zip(leaves, gradient_leaves, strict=True)
        ),
    )
    dot = tree_dot(gradient_tree, vector)
    hvp_leaves = torch.autograd.grad(dot, leaves, allow_unused=True)

    return tree_from_leaves(
        active,
        tuple(
            torch.zeros_like(leaf) if hvp is None else hvp.detach()
            for leaf, hvp in zip(leaves, hvp_leaves, strict=True)
        ),
    )


def hvp_jvp_grad_anchor(
    function: ScalarTensorFn,
    params: TensorTree,
    vector: TensorTree,
) -> Any:
    """Return HVP by JVP of grad."""
    return jvp_anchor(grad(function), params, vector)


def vhp_anchor(
    function: ScalarTensorFn,
    params: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Return VHP with PyTorch functional API."""
    _, result = torch.autograd.functional.vhp(function, params, vector)

    return result


def dense_jacobian_anchor(function: TensorFn, params: torch.Tensor) -> torch.Tensor:
    """Return a dense output-by-parameter Jacobian."""
    output = function(params)
    jacobian = torch.autograd.functional.jacobian(function, params)

    return jacobian.reshape(output.numel(), params.numel())


def ggnvp_dense_anchor(
    function: TensorFn,
    loss_hessian: torch.Tensor,
    params: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Return dense GGNVP as $J^T H Jv$."""
    jacobian = dense_jacobian_anchor(function, params)
    flat_vector = vector.reshape(-1)
    product = jacobian.T @ (loss_hessian @ (jacobian @ flat_vector))

    return product.reshape_as(params)


def fisher_vp_dense_anchor(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    *,
    normalization: float,
) -> torch.Tensor:
    """Return dense score-gradient outer-product product."""
    flat_vector = vector.reshape(-1)
    product = score_gradients.T @ (score_gradients @ flat_vector)

    return (product / normalization).reshape_as(vector)


def empirical_fisher_vp_dense_anchor(
    per_example_gradients: torch.Tensor,
    vector: torch.Tensor,
    *,
    normalization: float,
) -> torch.Tensor:
    """Return dense empirical Fisher-vector product."""
    flat_vector = vector.reshape(-1)
    product = per_example_gradients.T @ (per_example_gradients @ flat_vector)

    return (product / normalization).reshape_as(vector)


def dense_metric_multiply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Return dense metric-vector product."""
    product = matrix @ vector.reshape(-1)

    return product.reshape_as(vector)


def dense_metric_inverse_multiply(
    matrix: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Return dense metric inverse-vector product."""
    solution = torch.linalg.solve(matrix, vector.reshape(-1))

    return solution.reshape_as(vector)


def dense_metric_inner(
    matrix: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return dense metric inner product."""
    return left.reshape(-1) @ (matrix @ right.reshape(-1))


def dense_metric_inverse_residual(
    matrix: torch.Tensor,
    inverse_product: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Return relative inverse residual."""
    residual = dense_metric_multiply(matrix, inverse_product) - vector
    denominator = vector.reshape(-1).norm()

    if bool(torch.equal(denominator, torch.zeros_like(denominator))):
        return residual.reshape(-1).norm()

    return residual.reshape(-1).norm() / denominator


def finite_difference_jvp(
    function: TensorFn,
    params: TensorTree,
    vector: TensorTree,
    *,
    epsilon: float = 1e-5,
) -> Any:
    """Return central finite-difference JVP."""
    plus = function(
        tree_map2(lambda param, tangent: param + epsilon * tangent, params, vector)
    )
    minus = function(
        tree_map2(lambda param, tangent: param - epsilon * tangent, params, vector)
    )

    return tree_mul(tree_sub(plus, minus), 1.0 / (2.0 * epsilon))


def finite_difference_hvp(
    function: ScalarTensorFn,
    params: TensorTree,
    vector: TensorTree,
    *,
    epsilon: float = 1e-5,
) -> Any:
    """Return central finite-difference HVP."""
    plus_params = tree_map2(
        lambda param, tangent: param + epsilon * tangent, params, vector
    )
    minus_params = tree_map2(
        lambda param, tangent: param - epsilon * tangent, params, vector
    )
    plus = gradient_anchor(function, plus_params)
    minus = gradient_anchor(function, minus_params)

    return tree_mul(tree_sub(plus, minus), 1.0 / (2.0 * epsilon))


def vjp_dot_identity_error(
    function: TensorFn,
    params: TensorTree,
    tangent: TensorTree,
    cotangent: TensorTree,
) -> torch.Tensor:
    """Return absolute error in the JVP/VJP dot identity."""
    jvp_value = jvp_anchor(function, params, tangent)
    vjp_value = vjp_anchor(function, params, cotangent)
    left = tree_dot(jvp_value, cotangent)
    right = tree_dot(tangent, vjp_value)

    return (left - right).abs()


def _active_tree(params: TensorTree) -> TensorTree:
    return tree_map(lambda tensor: tensor.detach().clone().requires_grad_(True), params)


def module_functional_call(
    module: torch.nn.Module,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    *args: object,
    module_mode: str,
    tie_weights: bool,
    strict: bool,
    parametrization_policy: str,
    mutates_state: bool,
    mutated_parameter_keys: tuple[str, ...],
    mutated_buffer_keys: tuple[str, ...],
    **kwargs: object,
) -> object:
    """Call a module with explicit params and buffers.

    Returns:
        Module output.
    """
    admit_functional_call({
        "parameter_keys": tuple(params),
        "buffer_keys": tuple(buffers),
        "tie_weights": tie_weights,
        "strict": strict,
        "parametrization_policy": parametrization_policy,
        "mutates_state": mutates_state,
        "mutated_parameter_keys": mutated_parameter_keys,
        "mutated_buffer_keys": mutated_buffer_keys,
        "module_mode": module_mode,
    })
    was_training = module.training
    param_snapshots = _snapshot_selected(params, mutated_parameter_keys)
    buffer_snapshots = _snapshot_selected(buffers, mutated_buffer_keys)
    module_param_snapshots = _snapshot_selected(
        dict(module.named_parameters()),
        mutated_parameter_keys,
    )
    module_buffer_snapshots = _snapshot_selected(
        dict(module.named_buffers()),
        mutated_buffer_keys,
    )
    module.train(module_mode == "train")

    try:
        return functional_call(
            module,
            (params, buffers),
            args,
            kwargs,
            tie_weights=tie_weights,
            strict=strict,
        )
    finally:
        _restore_selected(params, param_snapshots)
        _restore_selected(buffers, buffer_snapshots)
        _restore_selected(dict(module.named_parameters()), module_param_snapshots)
        _restore_selected(dict(module.named_buffers()), module_buffer_snapshots)
        module.train(was_training)


def _snapshot_selected(
    tensors: dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {key: tensors[key].detach().clone() for key in keys if key in tensors}


def _restore_selected(
    tensors: dict[str, torch.Tensor],
    snapshots: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for key, snapshot in snapshots.items():
            tensors[key].copy_(snapshot)
