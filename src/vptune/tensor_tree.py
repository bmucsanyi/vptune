"""Tensor-tree utilities."""

from collections.abc import Callable, Sequence
from typing import Any, TypeGuard

import torch

TensorTree = torch.Tensor | tuple["TensorTree", ...] | dict[str, "TensorTree"]


def _is_tree_dict(tree: TensorTree) -> TypeGuard[dict[str, TensorTree]]:
    return isinstance(tree, dict)


def _is_tree_tuple(tree: TensorTree) -> TypeGuard[tuple[TensorTree, ...]]:
    return isinstance(tree, tuple)


def tree_map(
    fn: Callable[[torch.Tensor], torch.Tensor], tree: TensorTree
) -> TensorTree:
    """Apply a function to every tensor leaf.

    Returns:
        Mapped tensor tree.

    Raises:
        TypeError: If a leaf is not a tensor tree node.
    """
    if isinstance(tree, torch.Tensor):
        return fn(tree)

    if _is_tree_dict(tree):
        return {key: tree_map(fn, value) for key, value in tree.items()}

    if _is_tree_tuple(tree):
        return tuple(tree_map(fn, value) for value in tree)

    message = f"unsupported tensor tree leaf: {type(tree).__name__}"
    raise TypeError(message)


def tree_map2(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    """Apply a binary function to matching tensor leaves.

    Returns:
        Mapped tensor tree.

    Raises:
        RuntimeError: If matching container sizes differ.
        TypeError: If tree structures differ.
    """
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            message = "tensor tree tensor shapes differ"
            raise RuntimeError(message)

        return fn(left, right)

    if _is_tree_dict(left) and _is_tree_dict(right):
        if set(left) != set(right):
            message = "tensor tree mapping keys differ"
            raise RuntimeError(message)

        return {key: tree_map2(fn, left[key], right[key]) for key in left}

    if _is_tree_tuple(left) and _is_tree_tuple(right):
        if len(left) != len(right):
            message = "tensor tree sequence lengths differ"
            raise RuntimeError(message)

        return tuple(
            tree_map2(fn, lval, rval) for lval, rval in zip(left, right, strict=True)
        )

    message = "tensor tree structures differ"
    raise TypeError(message)


def tree_leaves(tree: TensorTree) -> tuple[torch.Tensor, ...]:
    """Return tensor leaves in deterministic order.

    Raises:
        TypeError: If a leaf is not a tensor tree node.
    """
    if isinstance(tree, torch.Tensor):
        return (tree,)

    if _is_tree_dict(tree):
        leaves = []

        for key in tree:
            leaves.extend(tree_leaves(tree[key]))

        return tuple(leaves)

    if _is_tree_tuple(tree):
        leaves = []

        for value in tree:
            leaves.extend(tree_leaves(value))

        return tuple(leaves)

    message = f"unsupported tensor tree leaf: {type(tree).__name__}"
    raise TypeError(message)


def tree_from_leaves(
    template: TensorTree,
    leaves: tuple[torch.Tensor, ...],
) -> TensorTree:
    """Return a tensor tree with the template structure and supplied leaves.

    Raises:
        RuntimeError: If too many leaves are supplied.
    """
    index = 0

    def build(node: TensorTree) -> TensorTree:
        nonlocal index

        if isinstance(node, torch.Tensor):
            if index >= len(leaves):
                message = "too few tensor leaves supplied"
                raise RuntimeError(message)

            leaf = leaves[index]
            index += 1

            return leaf

        if _is_tree_dict(node):
            return {key: build(node[key]) for key in node}

        if _is_tree_tuple(node):
            return tuple(build(value) for value in node)

        message = f"unsupported tensor tree leaf: {type(node).__name__}"
        raise TypeError(message)

    result = build(template)

    if index != len(leaves):
        message = "too many tensor leaves supplied"
        raise RuntimeError(message)

    return result


def tree_detach(tree: TensorTree) -> TensorTree:
    """Detach every tensor leaf.

    Returns:
        Detached tensor tree.
    """
    return tree_map(lambda tensor: tensor.detach(), tree)


def tree_zeros_like(tree: TensorTree) -> TensorTree:
    """Return a zero tree with matching leaves."""
    return tree_map(torch.zeros_like, tree)


def tree_add(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise sum."""
    return tree_map2(torch.add, left, right)


def tree_add_foreach(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise sum using PyTorch foreach kernels."""
    left_leaves, right_leaves = _matching_leaf_pairs(left, right)

    return _binary_foreach_tree(
        _torch_foreach("_foreach_add"),
        left,
        left_leaves,
        right_leaves,
    )


def tree_sub(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise difference."""
    return tree_map2(torch.sub, left, right)


def tree_mul(tree: TensorTree, scalar: float) -> TensorTree:
    """Return a tree scaled by a scalar."""
    return tree_map(lambda tensor: tensor * scalar, tree)


def tree_mul_foreach(tree: TensorTree, scalar: float) -> TensorTree:
    """Return a tree scaled by a scalar using PyTorch foreach kernels."""
    leaves = tree_leaves(tree)
    results = list(leaves)

    for indices, group in _foreach_groups(leaves):
        values = _torch_foreach("_foreach_mul")(group, scalar)
        _write_foreach_results(results, indices, values)

    return tree_from_leaves(tree, tuple(results))


def tree_add_scalar_foreach(tree: TensorTree, scalar: float) -> TensorTree:
    """Return a tree with a scalar added to every leaf using foreach kernels."""
    leaves = tree_leaves(tree)
    results = list(leaves)

    for indices, group in _foreach_groups(leaves):
        values = _torch_foreach("_foreach_add")(group, scalar)
        _write_foreach_results(results, indices, values)

    return tree_from_leaves(tree, tuple(results))


def tree_elementwise_mul_foreach(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise product using PyTorch foreach kernels."""
    left_leaves, right_leaves = _matching_leaf_pairs(left, right)

    return _binary_foreach_tree(
        _torch_foreach("_foreach_mul"),
        left,
        left_leaves,
        right_leaves,
    )


def tree_elementwise_div_foreach(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise quotient using PyTorch foreach kernels."""
    left_leaves, right_leaves = _matching_leaf_pairs(left, right)

    return _binary_foreach_tree(
        _torch_foreach("_foreach_div"),
        left,
        left_leaves,
        right_leaves,
    )


def tree_dot(left: TensorTree, right: TensorTree) -> torch.Tensor:
    """Return dot product over matching leaves."""
    products = tree_map2(
        lambda lval, rval: (lval.reshape(-1) * rval.reshape(-1)).sum(),
        left,
        right,
    )
    leaves = tree_leaves(products)

    if not leaves:
        return torch.tensor(0.0)

    total = leaves[0]

    for value in leaves[1:]:
        total = total + value

    return total


def tree_dot_foreach(left: TensorTree, right: TensorTree) -> torch.Tensor:
    """Return dot product using foreach elementwise products."""
    products = tree_elementwise_mul_foreach(left, right)
    leaves = tree_leaves(products)

    if not leaves:
        return torch.tensor(0.0)

    total = leaves[0].reshape(-1).sum()

    for value in leaves[1:]:
        total = total + value.reshape(-1).sum()

    return total


def tree_max_abs(tree: TensorTree) -> torch.Tensor:
    """Return max absolute value across leaves."""
    leaves = tree_leaves(tree)

    if not leaves:
        return torch.tensor(0.0)

    values = tuple(tensor.detach().abs().max() for tensor in leaves)
    result = values[0]

    for value in values[1:]:
        result = torch.maximum(result, value)

    return result


def tree_l2_norm(tree: TensorTree) -> torch.Tensor:
    """Return L2 norm across leaves."""
    return torch.sqrt(tree_dot(tree, tree))


def tree_signature(tree: TensorTree) -> dict[str, Any]:
    """Return shape, dtype, and key structure for a tensor tree.

    Raises:
        TypeError: If a leaf is not a tensor tree node.
    """
    if isinstance(tree, torch.Tensor):
        return {
            "shape": tuple(tree.shape),
            "dtype": str(tree.dtype).removeprefix("torch."),
            "device": str(tree.device),
        }

    if _is_tree_dict(tree):
        return {
            "type": "mapping",
            "items": tuple(
                {"key": str(key), "value": tree_signature(value)}
                for key, value in tree.items()
            ),
        }

    if _is_tree_tuple(tree):
        return {
            "type": "sequence",
            "items": tuple(tree_signature(value) for value in tree),
        }

    message = f"unsupported tensor tree leaf: {type(tree).__name__}"
    raise TypeError(message)


def _matching_leaf_pairs(
    left: TensorTree,
    right: TensorTree,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    checked_left = tree_map2(lambda lval, _: lval, left, right)
    checked_right = tree_map2(lambda _, rval: rval, left, right)

    return tree_leaves(checked_left), tree_leaves(checked_right)


def _foreach_groups(
    leaves: tuple[torch.Tensor, ...],
) -> tuple[tuple[tuple[int, ...], tuple[torch.Tensor, ...]], ...]:
    groups = {}

    for index, leaf in enumerate(leaves):
        key = (leaf.device, leaf.dtype)

        if key not in groups:
            groups[key] = ([], [])

        groups[key][0].append(index)
        groups[key][1].append(leaf)

    return tuple((tuple(indices), tuple(group)) for indices, group in groups.values())


def _foreach_binary_groups(
    left_leaves: tuple[torch.Tensor, ...],
    right_leaves: tuple[torch.Tensor, ...],
) -> tuple[
    tuple[tuple[int, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], ...
]:
    groups = {}

    for index, (left, right) in enumerate(zip(left_leaves, right_leaves, strict=True)):
        if left.device != right.device:
            message = "foreach tensor pairs must be on the same device"
            raise RuntimeError(message)

        if left.dtype != right.dtype:
            message = "foreach tensor pairs must have the same dtype"
            raise RuntimeError(message)

        key = (left.device, left.dtype)

        if key not in groups:
            groups[key] = ([], [], [])

        groups[key][0].append(index)
        groups[key][1].append(left)
        groups[key][2].append(right)

    return tuple(
        (tuple(indices), tuple(left_group), tuple(right_group))
        for indices, left_group, right_group in groups.values()
    )


def _binary_foreach_tree(
    operation: Any,
    template: TensorTree,
    left_leaves: tuple[torch.Tensor, ...],
    right_leaves: tuple[torch.Tensor, ...],
) -> TensorTree:
    results = list(left_leaves)

    for indices, left_group, right_group in _foreach_binary_groups(
        left_leaves,
        right_leaves,
    ):
        values = operation(left_group, right_group)
        _write_foreach_results(results, indices, values)

    return tree_from_leaves(template, tuple(results))


def _write_foreach_results(
    results: list[torch.Tensor],
    indices: tuple[int, ...],
    values: Sequence[torch.Tensor],
) -> None:
    for index, value in zip(indices, values, strict=True):
        results[index] = value


def _torch_foreach(name: str) -> Any:
    return getattr(torch, name)
