# vptune

Autotuning for matrix-free derivative-vector products in PyTorch.

## Install

`vptune` requires Python 3.12 through 3.14, NumPy, and PyTorch 2.12.x.

From a checkout:

```bash
python -m pip install .
```

For development:

```bash
uv sync --extra dev
make lint-fix
make test
```

## Quickstart

This example tunes a parameter-gradient product on CPU. For an H100 run, replace
`cpu_target()` with `vp.cuda(0, "h100")`.

```python
from pathlib import Path
import tempfile

import torch

import vptune as vp


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": x @ self.weight.T}


def cpu_target() -> vp.Target:
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


module = TinyClassifier()
model = vp.torch_model(
    module,
    parameters=vp.parameters(module),
    call=vp.module_call(args=("x",), kwargs={}, output="logits"),
)
loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
product = vp.gradient(model, loss, name="loss_gradient")
batch = {
    "x": torch.tensor([[1.0, -0.5], [0.25, 2.0]], dtype=torch.float64),
    "labels": torch.tensor([0, 1], dtype=torch.long),
}
vector = {
    "weight": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
}

with tempfile.TemporaryDirectory() as directory:
    tuned = product.tune(
        data=(batch,),
        vectors=(vector,),
        target=cpu_target(),
        space=vp.space.standard(),
        search=vp.search.exhaustive(),
        run_dir=Path(directory),
    )
    gradient = tuned(batch)

print(gradient["weight"])
```

## Public API boundary

Normal callers import `vptune as vp` and use typed objects such as
`vp.torch_model`, `vp.loss.*`, `vp.gradient`, `vp.ggnvp`, `vp.metric.*`,
`vp.space.*`, `vp.search.*`, `vp.cuda`, and `vp.tune`.

Adapter authors and custom runtime authors use `vptune.ext`. Candidate rows,
runtime configs, manifests, package anchors, memory backends, and schema helpers
stay out of the root namespace.

Custom scalar and function objectives also use `vptune.ext` protocol types for
their callback signatures.
