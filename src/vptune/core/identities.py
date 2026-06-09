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

ENVIRONMENT_IDENTITY_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "CUBLAS_WORKSPACE_CONFIG",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
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


def tensor_signature(
    tensor: torch.Tensor,
    *,
    include_requires_grad: bool = True,
) -> dict[str, Any]:
    """Return identity fields for a tensor without copying its values."""
    if include_requires_grad:
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
        }

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
    }


def module_identity(model: torch.nn.Module) -> dict[str, Any]:
    """Return stable identity fields for a module instance."""
    named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    parameters = tuple(
        {
            "name": name,
            **tensor_signature(parameter),
        }
        for name, parameter in named_parameters
    )
    buffers = tuple(
        {
            "name": name,
            **tensor_signature(buffer, include_requires_grad=False),
        }
        for name, buffer in model.named_buffers()
    )

    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "training": model.training,
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


def parametrization_names(model: torch.nn.Module) -> tuple[dict[str, Any], ...]:
    """Return active parametrization identities in module traversal order."""
    records = []

    for module_name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)

        if parametrizations is None:
            continue

        if not hasattr(parametrizations, "keys"):
            continue

        for name in sorted(parametrizations.keys()):
            parameter_name = str(name)
            prefix = f"{module_name}." if module_name else ""
            parametrization_list = getattr(parametrizations, parameter_name)
            records.append({
                "parameter": f"{prefix}{parameter_name}",
                "parametrizations": tuple(
                    f"{type(item).__module__}.{type(item).__qualname__}"
                    for item in parametrization_list
                ),
            })

    return tuple(records)


def environment_signature() -> dict[str, Any]:
    """Return runtime fields that affect measurement and replay."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_signature(),
        "determinism": determinism_signature(),
        "backend_flags": backend_flags_signature(),
        "cuda": cuda_environment_signature(),
        "rocm": rocm_environment_signature(),
        "mps": mps_environment_signature(),
        "env": relevant_environment_variables(),
    }


def torch_signature() -> dict[str, Any]:
    """Return PyTorch build identity."""
    return {
        "version": torch.__version__,
        "config": torch.__config__.show(),
    }


def determinism_signature() -> dict[str, Any]:
    """Return PyTorch determinism flags."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
    }


def backend_flags_signature() -> dict[str, Any]:
    """Return backend settings that change numerical kernels."""
    return {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def cuda_environment_signature() -> dict[str, Any]:
    """Return CUDA runtime and visible CUDA device identity."""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "runtime": torch.version.cuda,
            "driver_version": None,
            "device_count": 0,
            "devices": (),
        }

    return {
        "available": True,
        "runtime": torch.version.cuda,
        "driver_version": cuda_driver_version(),
        "device_count": torch.cuda.device_count(),
        "devices": tuple(
            cuda_device_signature(torch.device("cuda", index))
            for index in range(torch.cuda.device_count())
        ),
    }


def cuda_driver_version() -> Any:
    """Return the CUDA driver version reported by the CUDA runtime.

    Raises:
        RuntimeError: If the CUDA runtime reports a failed driver-version query.
    """
    driver_version = getattr(torch.cuda.cudart(), "cudaDriverGetVersion", None)

    if driver_version is None:
        return None

    version = driver_version()

    if isinstance(version, tuple):
        error_code, driver_version = version

        if error_code != 0:
            message = f"CUDA driver version query failed with code {error_code}"
            raise RuntimeError(message)

        return driver_version

    return version


def rocm_environment_signature() -> dict[str, Any]:
    """Return ROCm runtime identity."""
    return {
        "runtime": torch.version.hip,
        "available": torch.version.hip is not None and torch.cuda.is_available(),
    }


def mps_environment_signature() -> dict[str, Any]:
    """Return MPS runtime identity."""
    return {
        "built": torch.backends.mps.is_built(),
        "available": torch.backends.mps.is_available(),
    }


def relevant_environment_variables() -> dict[str, str]:
    """Return environment variables that affect compilation or device execution."""
    return {
        key: os.environ[key] for key in ENVIRONMENT_IDENTITY_KEYS if key in os.environ
    }


def device_signature(device: str | torch.device) -> dict[str, Any]:
    """Return hardware identity for one declared target device.

    Raises:
        RuntimeError: If a CUDA device is declared while CUDA is unavailable.
    """
    torch_device = torch.device(device)

    if torch_device.type == "cpu":
        return {
            "device": str(torch_device),
            "type": "cpu",
            "index": torch_device.index,
        }

    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            message = f"CUDA target device is unavailable: {torch_device}"
            raise RuntimeError(message)

        return cuda_device_signature(torch_device)

    if torch_device.type == "mps":
        return {
            "device": str(torch_device),
            "type": "mps",
            "index": torch_device.index,
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        }

    return {
        "device": str(torch_device),
        "type": torch_device.type,
        "index": torch_device.index,
    }


def cuda_device_signature(device: torch.device) -> dict[str, Any]:
    """Return identity for one CUDA or ROCm device."""
    index = device.index

    if index is None:
        index = torch.cuda.current_device()

    properties = torch.cuda.get_device_properties(index)

    return {
        "device": str(torch.device("cuda", index)),
        "type": "cuda",
        "runtime": "rocm" if torch.version.hip is not None else "cuda",
        "index": index,
        "name": properties.name,
        "capability": (properties.major, properties.minor),
        "total_memory": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
    }
