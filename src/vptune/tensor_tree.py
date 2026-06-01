"""Tensor-tree utilities."""

from typing import Any, TypeGuard

import torch

TensorTree = torch.Tensor | tuple["TensorTree", ...] | dict[str, "TensorTree"]


def _is_tree_dict(tree: TensorTree) -> TypeGuard[dict[str, TensorTree]]:
    return isinstance(tree, dict)


def _is_tree_tuple(tree: TensorTree) -> TypeGuard[tuple[TensorTree, ...]]:
    return isinstance(tree, tuple)


def tree_map(fn: Any, tree: TensorTree) -> TensorTree:
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


def tree_map2(fn: Any, left: TensorTree, right: TensorTree) -> TensorTree:
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


def tree_sub(left: TensorTree, right: TensorTree) -> TensorTree:
    """Return tree-wise difference."""
    return tree_map2(torch.sub, left, right)


def tree_mul(tree: TensorTree, scalar: float) -> TensorTree:
    """Return a tree scaled by a scalar."""
    return tree_map(lambda tensor: tensor * scalar, tree)


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
