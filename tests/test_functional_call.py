from typing import override

import pytest
import torch
from torch.nn.utils import parametrize

import vptune as vp


def functional_call_settings(**overrides: object) -> dict[str, object]:
    return {
        "parameter_keys": overrides.get("parameter_keys", ("weight",)),
        "buffer_keys": overrides.get("buffer_keys", ("scale",)),
        "tie_weights": overrides.get("tie_weights", True),
        "strict": overrides.get("strict", False),
        "parametrization_policy": overrides.get("parametrization_policy", "active"),
        "mutates_state": overrides.get("mutates_state", False),
        "mutated_parameter_keys": overrides.get("mutated_parameter_keys", ()),
        "mutated_buffer_keys": overrides.get("mutated_buffer_keys", ()),
        "module_mode": overrides.get("module_mode", "eval"),
    }


def test_functional_call_admission_validates_declared_fields() -> None:
    vp.admit_functional_call(functional_call_settings())

    rejected = (
        {"parameter_keys": ["weight"]},
        {"buffer_keys": ["scale"]},
        {"tie_weights": 1},
        {"strict": 0},
        {"parametrization_policy": "unknown"},
        {"mutates_state": True},
        {"mutated_parameter_keys": ["weight"]},
        {"mutated_buffer_keys": ["scale"]},
        {"mutated_parameter_keys": ("other",), "mutates_state": True},
        {"mutated_buffer_keys": ("other",), "mutates_state": True},
        {"mutated_buffer_keys": ("scale",)},
        {"module_mode": "unknown"},
    )

    for settings_override in rejected:
        with pytest.raises(vp.AdmissionError):
            vp.admit_functional_call(functional_call_settings(**settings_override))


def test_module_functional_call_handles_buffers_and_restores_mode() -> None:
    class BufferModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.register_buffer("scale", torch.tensor([2.0]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            mode_offset = torch.tensor([10.0]) if self.training else torch.tensor([0.0])

            return value * self.weight * self.get_buffer("scale") + mode_offset

    module = BufferModule()
    module.train()
    output = vp.module_functional_call(
        module,
        {"weight": torch.tensor([3.0])},
        {"scale": torch.tensor([4.0])},
        torch.tensor([2.0]),
        module_mode="eval",
        tie_weights=True,
        strict=False,
        parametrization_policy="active",
        mutates_state=False,
        mutated_parameter_keys=(),
        mutated_buffer_keys=(),
    )

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.tensor([24.0]))
    assert module.training is True
    assert torch.equal(module.weight, torch.tensor([1.0]))
    assert torch.equal(dict(module.named_buffers())["scale"], torch.tensor([2.0]))


def test_module_functional_call_respects_tied_weight_policy() -> None:
    class TiedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.nn.Parameter(torch.tensor([1.0]))
            self.left = shared
            self.right = shared

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.left * value + self.right * value

    module = TiedModule()
    tied = vp.module_functional_call(
        module,
        {"left": torch.tensor([2.0])},
        {},
        torch.tensor([1.0]),
        module_mode="eval",
        tie_weights=True,
        strict=False,
        parametrization_policy="active",
        mutates_state=False,
        mutated_parameter_keys=(),
        mutated_buffer_keys=(),
    )
    untied = vp.module_functional_call(
        module,
        {"left": torch.tensor([2.0])},
        {},
        torch.tensor([1.0]),
        module_mode="eval",
        tie_weights=False,
        strict=False,
        parametrization_policy="active",
        mutates_state=False,
        mutated_parameter_keys=(),
        mutated_buffer_keys=(),
    )

    assert isinstance(tied, torch.Tensor)
    assert isinstance(untied, torch.Tensor)
    assert torch.equal(tied, torch.tensor([4.0]))
    assert torch.equal(untied, torch.tensor([3.0]))


def test_module_functional_call_preserves_active_parametrization() -> None:
    class ExpParametrization(torch.nn.Module):
        @override
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.exp()

    module = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    parametrize.register_parametrization(
        module,
        "weight",
        ExpParametrization(),
    )
    output = vp.module_functional_call(
        module,
        {"parametrizations.weight.original": torch.zeros(1, 1, dtype=torch.float64)},
        {},
        torch.ones(1, 1, dtype=torch.float64),
        module_mode="eval",
        tie_weights=True,
        strict=False,
        parametrization_policy="active",
        mutates_state=False,
        mutated_parameter_keys=(),
        mutated_buffer_keys=(),
    )

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.ones(1, 1, dtype=torch.float64))


def test_module_functional_call_resets_declared_buffer_mutation() -> None:
    class MutatingBufferModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("count", torch.tensor([0.0]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            count = self.get_buffer("count")
            count.add_(1.0)

            return value + count

    module = MutatingBufferModule()
    buffers = {"count": torch.tensor([5.0])}
    output = vp.module_functional_call(
        module,
        {},
        buffers,
        torch.tensor([1.0]),
        module_mode="eval",
        tie_weights=True,
        strict=False,
        parametrization_policy="active",
        mutates_state=True,
        mutated_parameter_keys=(),
        mutated_buffer_keys=("count",),
    )

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.tensor([7.0]))
    assert torch.equal(buffers["count"], torch.tensor([5.0]))
    assert torch.equal(dict(module.named_buffers())["count"], torch.tensor([0.0]))


def test_module_functional_call_resets_declared_mutation_after_error() -> None:
    class FailingBufferModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("count", torch.tensor([0.0]))

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            count = self.get_buffer("count")
            count.add_(1.0)
            message = "boom"
            raise RuntimeError(message)

    module = FailingBufferModule()
    buffers = {"count": torch.tensor([5.0])}

    with pytest.raises(RuntimeError, match="boom"):
        vp.module_functional_call(
            module,
            {},
            buffers,
            torch.tensor([1.0]),
            module_mode="eval",
            tie_weights=True,
            strict=False,
            parametrization_policy="active",
            mutates_state=True,
            mutated_parameter_keys=(),
            mutated_buffer_keys=("count",),
        )

    assert torch.equal(buffers["count"], torch.tensor([5.0]))
    assert torch.equal(dict(module.named_buffers())["count"], torch.tensor([0.0]))
