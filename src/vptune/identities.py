"""Stable identity and hashing helpers."""

import dataclasses
import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from typing import Any

import torch

JsonValue = (
    None | bool | int | float | str | tuple["JsonValue", ...] | dict[str, "JsonValue"]
)


def to_json_value(value: Any) -> JsonValue:
    """Return a deterministic JSON-compatible value.

    Raises:
        TypeError: If the value cannot be represented as JSON.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if dataclasses.is_dataclass(value):
        return to_json_value(dataclasses.asdict(value))

    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): to_json_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(to_json_value(item) for item in value)

    message = f"value is not JSON-compatible: {type(value).__name__}"
    raise TypeError(message)


def canonical_json(value: Any) -> str:
    """Return canonical JSON for hashing and record comparison."""
    json_value = to_json_value(value)

    return json.dumps(json_value, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    """Return a stable SHA256 hash for a JSON-compatible value."""
    payload = canonical_json(value).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def owner_hash(scope: str, payload: Mapping[str, Any]) -> str:
    """Return an owner hash with an explicit scope field."""
    scoped = {"scope": scope, "payload": dict(payload)}

    return stable_hash(scoped)


def record_content_hash(payload: Mapping[str, Any]) -> str:
    """Return a hash of saved row content."""
    content = dict(payload)
    content.pop("content_hash", None)

    return owner_hash("record_content", content)


def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Return identity fields for a tensor without copying its values."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def module_identity(model: torch.nn.Module) -> dict[str, Any]:
    """Return stable identity fields for a module instance."""
    named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    parameters = tuple(
        {
            "name": name,
            "shape": tuple(parameter.shape),
            "dtype": str(parameter.dtype).removeprefix("torch."),
            "device": str(parameter.device),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named_parameters
    )
    buffers = tuple(
        {
            "name": name,
            "shape": tuple(buffer.shape),
            "dtype": str(buffer.dtype).removeprefix("torch."),
            "device": str(buffer.device),
        }
        for name, buffer in model.named_buffers()
    )

    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "training": bool(model.training),
        "parameters": parameters,
        "buffers": buffers,
        "tied_parameter_groups": tied_parameter_groups(named_parameters),
        "parametrizations": parametrization_names(model),
    }


def tied_parameter_groups(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[tuple[str, ...], ...]:
    """Return groups of parameter names sharing the same tensor object."""
    names_by_id = {}

    for name, parameter in named_parameters:
        names_by_id.setdefault(id(parameter), []).append(name)

    groups = tuple(
        tuple(sorted(names)) for names in names_by_id.values() if len(names) > 1
    )

    return tuple(sorted(groups))


def parametrization_names(model: torch.nn.Module) -> tuple[str, ...]:
    """Return active parametrization names on a module."""
    parametrizations = getattr(model, "parametrizations", None)

    if parametrizations is None:
        return ()

    if not hasattr(parametrizations, "keys"):
        return ()

    return tuple(str(name) for name in sorted(parametrizations.keys()))


def environment_signature() -> dict[str, Any]:
    """Return runtime fields that affect measurement and replay."""
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "env": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "CUBLAS_WORKSPACE_CONFIG",
            )
            if key in os.environ
        },
    }

    if not torch.cuda.is_available():
        return payload

    return {
        **payload,
        "cuda": {
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": tuple(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
