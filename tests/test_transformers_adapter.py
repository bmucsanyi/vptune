import contextlib
import dataclasses
from collections.abc import Callable, Iterator, Mapping

import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
import vptune.adapters.transformers as transformers_module
import vptune.ext as vpx
import vptune.runtime as runtime_module
from vptune import operators as ops
from vptune.errors import AdmissionError


def transformers_policy(*, use_cache: bool = False) -> vpa.TransformersAttentionPolicy:
    return vpa.TransformersAttentionPolicy(
        model_config_hash="model",
        use_cache=use_cache,
        softcap={"logit_softcap": 30.0},
        mask_semantics="boolean_keep_mask",
        causal_policy="causal",
        backend_numeric_policy={"backend": "sdpa"},
        determinism={"deterministic": True},
        padding_limit=4096,
        forced_kernel_available=True,
    )


class TinyTiedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.embedding = shared
        self.lm_head = shared
        self.register_buffer("scale", torch.tensor([0.5]))


class TinyTransformersScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.attention_values = []

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.attention_values.append(attn_implementation)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.w * scale).sum()


class RowSelectedAttentionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.attention_values = []
        self.attention_factors = {
            "custom_attention_a": 2.0,
            "custom_attention_b": 3.0,
        }
        self.mask_values = []
        self.mask_factors = {
            "custom_attention_a": 5.0,
            "custom_attention_b": 7.0,
        }
        self.attention_factor = 1.0
        self.mask_factor = 1.0

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.attention_values.append(attn_implementation)
        self.attention_factor = self.attention_factors[attn_implementation]

    def set_attention_mask_formatter(self, mask_formatter_id: str) -> None:
        self.mask_values.append(mask_formatter_id)
        self.mask_factor = self.mask_factors[mask_formatter_id]

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.w * scale * self.attention_factor * self.mask_factor).sum()


class ContextCheckingTransformersModule(TinyTransformersScalarModule):
    def __init__(self, active: list[bool], forward_events: list[bool]) -> None:
        super().__init__()
        self.active = active
        self.forward_events = forward_events

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        self.forward_events.append(bool(self.active))

        return super().forward(scale)


def stateful_transformers_settings() -> dict[str, object]:
    return {
        "gradient.path": "torch_autograd_grad",
        "call.path": "stateful_module",
        "call.params": "module_params",
        "call.buffers": "module_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "forbidden",
        "call.grad_mode": "grad_enabled",
        "call.return_type": "raw_tensor_tree",
        "attention.frontend": "transformers_eager",
        "module_mode": "eval",
        "dropout_p": 0.0,
    }


def tensor_dict(tree: object) -> dict[str, torch.Tensor]:
    assert isinstance(tree, dict)
    result = {}

    for key, value in tree.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)
        result[key] = value

    return result


def test_transformers_operation_factory_sets_attention_and_runs_module() -> None:
    model = TinyTransformersScalarModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    output = factory(
        vpx.Candidate(
            "gradient",
            "transformers-gradient",
            stateful_transformers_settings(),
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()

    assert model.attention_values == ["eager"]
    assert not model.training
    output_map = tensor_dict(output)
    torch.testing.assert_close(
        output_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_transformers_registered_attention_row_selects_runtime_backend() -> None:
    model = RowSelectedAttentionModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    batch = {"scale": torch.tensor([4.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    first = factory(
        vpx.Candidate(
            "gradient",
            "registered-a",
            {
                **stateful_transformers_settings(),
                "attention.frontend": "registered_transformers_attention",
                "attention.custom_kernel_id": "custom_attention_a",
                "attention.mask_formatter_id": "custom_attention_a",
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()
    second = factory(
        vpx.Candidate(
            "gradient",
            "registered-b",
            {
                **stateful_transformers_settings(),
                "attention.frontend": "registered_transformers_attention",
                "attention.custom_kernel_id": "custom_attention_b",
                "attention.mask_formatter_id": "custom_attention_b",
            },
            admission_status="passed",
        ),
        batch,
        vector,
    )()

    assert model.attention_values == ["custom_attention_a", "custom_attention_b"]
    assert model.mask_values == ["custom_attention_a", "custom_attention_b"]
    torch.testing.assert_close(
        tensor_dict(first)["w"],
        torch.tensor([40.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tensor_dict(second)["w"],
        torch.tensor([84.0], dtype=torch.float64),
    )


def test_transformers_registered_attention_requires_mask_formatter_runtime() -> None:
    model = TinyTransformersScalarModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )

    with pytest.raises(vp.AdmissionError, match="set_attention_mask_formatter"):
        factory(
            vpx.Candidate(
                "gradient",
                "registered",
                {
                    **stateful_transformers_settings(),
                    "attention.frontend": "registered_transformers_attention",
                    "attention.custom_kernel_id": "custom_attention_a",
                    "attention.mask_formatter_id": "custom_attention_a",
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor([4.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_transformers_registered_attention_rejects_conflicting_runtime_id() -> None:
    model = RowSelectedAttentionModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        attention_custom_kernel_id="custom_attention_a",
    )

    with pytest.raises(vp.AdmissionError, match="must match"):
        factory(
            vpx.Candidate(
                "gradient",
                "registered-b",
                {
                    **stateful_transformers_settings(),
                    "attention.frontend": "registered_transformers_attention",
                    "attention.custom_kernel_id": "custom_attention_b",
                    "attention.mask_formatter_id": "custom_attention_b",
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor([4.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


@pytest.mark.parametrize(
    ("extra_setting", "message"),
    [
        ({"use_cache": True}, "model-load setting"),
        ({"output_attentions": True}, "attention-weights output surface"),
    ],
)
def test_transformers_runtime_rejects_unlowered_row_settings(
    extra_setting: dict[str, object],
    message: str,
) -> None:
    model = TinyTransformersScalarModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )

    with pytest.raises(vp.AdmissionError, match=message):
        factory(
            vpx.Candidate(
                "gradient",
                "unlowered-transformers-row",
                {
                    **stateful_transformers_settings(),
                    **extra_setting,
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor([4.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_transformers_sdpa_rows_enter_declared_kernel_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = []
    forward_events = []
    kernel_calls = []
    model = ContextCheckingTransformersModule(active, forward_events)
    settings = {
        **stateful_transformers_settings(),
        "attention.frontend": "transformers_sdpa",
        "attention.sdpa_kernel": "priority_list",
        "attention.sdpa_priority_list": ("flash_attention", "math"),
    }
    candidate = vpx.Candidate(
        "gradient",
        "transformers-sdpa",
        settings,
        admission_status="passed",
    )

    @contextlib.contextmanager
    def fake_sdpa_kernel(
        backends: object,
        *,
        set_priority: bool,
    ) -> Iterator[None]:
        assert isinstance(backends, list)
        kernel_calls.append((tuple(backends), set_priority))
        active.append(True)

        try:
            yield
        finally:
            active.pop()

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    monkeypatch.setattr(transformers_module, "sdpa_kernel", fake_sdpa_kernel)
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    check = vpa.transformers_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar_objective},
    )
    batch = {"scale": torch.tensor([4.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}

    output = factory(candidate, batch, vector)()
    reference = check(candidate, batch, vector)

    assert model.attention_values == ["sdpa", "sdpa"]
    assert all(forward_events)
    assert len(kernel_calls) == 3
    assert kernel_calls
    assert {call[1] for call in kernel_calls} == {True}
    assert {call[0] for call in kernel_calls} == {
        (
            transformers_module.SDPA_KERNEL_BACKENDS["flash_attention"],
            transformers_module.SDPA_KERNEL_BACKENDS["math"],
        )
    }
    torch.testing.assert_close(
        tensor_dict(output)["w"],
        torch.tensor([4.0], dtype=torch.float64),
    )
    assert reference.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_transformers_sdpa_warm_compile_enters_declared_kernel_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = []
    forward_events = []
    kernel_calls = []
    model = ContextCheckingTransformersModule(active, forward_events)
    settings = {
        **stateful_transformers_settings(),
        "attention.frontend": "transformers_sdpa",
        "attention.sdpa_kernel": "math",
        "compile.enabled": "true",
        "compile.boundary": "whole_operator",
        "compile.backend": "inductor",
        "compile.mode": "default",
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": "false",
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": "false",
        "compile.cache_state": "warm_cache",
    }
    candidate = vpx.Candidate(
        "gradient",
        "transformers-sdpa-warm",
        settings,
        admission_status="passed",
    )

    @contextlib.contextmanager
    def fake_sdpa_kernel(
        backend: object,
        *,
        set_priority: bool,
    ) -> Iterator[None]:
        kernel_calls.append((backend, set_priority))
        active.append(True)

        try:
            yield
        finally:
            active.pop()

    def fake_compile(
        operation: Callable[[], vpx.TensorTree],
        **_: object,
    ) -> Callable[[], vpx.TensorTree]:
        return operation

    monkeypatch.setattr(transformers_module, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(runtime_module.torch, "compile", fake_compile)
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    batch = {"scale": torch.tensor([4.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    operation = factory(candidate, batch, vector)

    assert forward_events == [True]
    output = operation()

    assert forward_events == [True, True]
    assert kernel_calls == [
        (transformers_module.SDPA_KERNEL_BACKENDS["math"], False),
        (transformers_module.SDPA_KERNEL_BACKENDS["math"], False),
    ]
    torch.testing.assert_close(
        tensor_dict(output)["w"],
        torch.tensor([4.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("boundary", "block_paths", "attention_paths"),
    [
        ("transformer_block", ("block",), ()),
        ("attention_module", (), ("block",)),
    ],
)
def test_transformers_operation_factory_compiles_declared_submodule_boundary(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    block_paths: tuple[str, ...],
    attention_paths: tuple[str, ...],
) -> None:
    events = []

    class TinyBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))

        def forward(self, scale: torch.Tensor) -> torch.Tensor:
            events.append("block_forward")

            return self.w * scale

    class TinyBlockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = TinyBlock()
            self.attention_values = []

        def set_attn_implementation(self, attn_implementation: str) -> None:
            self.attention_values.append(attn_implementation)

        def forward(self, scale: torch.Tensor) -> torch.Tensor:
            return self.block(scale).sum()

    def fake_compile(
        forward: Callable[..., object],
        *,
        backend: str,
        mode: str | None,
        fullgraph: bool,
        dynamic: bool | None,
        options: Mapping[str, bool] | None,
    ) -> Callable[..., object]:
        events.append({
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "options": options,
        })

        def compiled(*args: object, **kwargs: object) -> object:
            events.append("compiled_forward")

            return forward(*args, **kwargs)

        return compiled

    monkeypatch.setattr(runtime_module.torch, "compile", fake_compile)
    model = TinyBlockModel()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        transformer_block_paths=block_paths,
        attention_module_paths=attention_paths,
    )
    operation = factory(
        vpx.Candidate(
            "gradient",
            "compiled-submodule",
            {
                **stateful_transformers_settings(),
                "compile.enabled": "true",
                "compile.boundary": boundary,
                "compile.backend": "inductor",
                "compile.mode": "default",
                "compile.fullgraph": "false",
                "compile.dynamic": None,
                "compile.compiled_autograd": "false",
                "compile.options.epilogue_fusion": "false",
                "compile.options.shape_padding": "false",
                "compile.cuda_graphs": "false",
                "compile.cache_state": "cold_compile",
            },
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"block.w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert events == [
        {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": None,
            "options": None,
        }
    ]
    output = operation()

    assert events == [
        {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": None,
            "options": None,
        },
        "compiled_forward",
        "block_forward",
    ]
    assert "forward" not in model.block.__dict__
    torch.testing.assert_close(
        tensor_dict(output)["block.w"],
        torch.tensor([4.0], dtype=torch.float64),
    )


def test_transformers_operation_factory_rejects_compile_boundary_without_paths() -> (
    None
):
    model = TinyTransformersScalarModule()
    factory = vpa.transformers_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
    )

    with pytest.raises(vp.MaterializationError, match="declared module paths"):
        factory(
            vpx.Candidate(
                "gradient",
                "missing-paths",
                {
                    **stateful_transformers_settings(),
                    "compile.enabled": "true",
                    "compile.boundary": "transformer_block",
                    "compile.backend": "inductor",
                    "compile.mode": "default",
                    "compile.fullgraph": "false",
                    "compile.dynamic": None,
                    "compile.compiled_autograd": "false",
                    "compile.options.epilogue_fusion": "false",
                    "compile.options.shape_padding": "false",
                    "compile.cuda_graphs": "false",
                    "compile.cache_state": "cold_compile",
                },
                admission_status="passed",
            ),
            {"scale": torch.tensor([4.0], dtype=torch.float64)},
            {"w": torch.tensor([1.0], dtype=torch.float64)},
        )


def test_transformers_reference_check_runs_independent_anchor() -> None:
    model = TinyTransformersScalarModule()

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    check = vpa.transformers_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar_objective},
    )
    result = check(
        vpx.Candidate(
            "gradient",
            "transformers-gradient",
            stateful_transformers_settings(),
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert model.attention_values == ["eager"]
    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_transformers_runtime_config_runs_full_size_check_and_materializer() -> None:
    model = TinyTransformersScalarModule()

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    candidate = vpx.Candidate(
        "gradient",
        "transformers-gradient",
        stateful_transformers_settings(),
        admission_status="passed",
    )
    runtime = vpa.transformers_runtime_config(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        candidates=(candidate,),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        objective_signature={"case": "transformers-runtime"},
        module_call=vpx.ModuleCallSpec(positional_batch_keys=("scale",)),
        axis_registry=None,
        scalar_objectives={"loss": scalar_objective},
    )
    batch = {"scale": torch.tensor([4.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    operation = runtime.operation_factory(candidate, batch, vector)
    output = operation()

    assert runtime.full_size_check is not None
    metadata = runtime.full_size_check(
        candidate,
        ((batch, vector),),
        (output,),
        (),
    )
    sample = vpx.Measurement(
        elapsed_seconds=1.0,
        peak_allocated_mib=1.0,
        peak_reserved_mib=1.0,
        post_allocated_mib=0.0,
        post_reserved_mib=0.0,
    )
    record = vpx.FullSizeRecord(
        family="gradient",
        candidate_id="transformers-gradient",
        status="passed",
        input_signature={},
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=(sample,),
        memory_samples=(sample,),
    )
    selected = runtime.materializer(candidate, record)
    selected_output = selected(batch, vector)
    selected_map = tensor_dict(selected_output)

    assert runtime.identity()["full_size_check"] is not None
    assert metadata["transformers_full_size_checked_inputs"] == 1
    torch.testing.assert_close(
        selected_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_transformers_attention_location_executes_core_attention() -> None:
    query = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 17.0
    key = torch.arange(24, 48, dtype=torch.float32).reshape(1, 2, 4, 3) / 19.0
    value = torch.arange(48, 72, dtype=torch.float32).reshape(1, 2, 4, 3) / 23.0
    location = vpa.transformers_attention_location(
        query_key="query",
        key_key="key",
        value_key="value",
        output_key="attention",
        mask_key=None,
        inverse_permutation_key=None,
        query_block_size_key=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        causal_policy="bidirectional",
        sliding_window_policy="none",
        padding_policy="dense",
        mask_convention="additive_or_bool",
        dropout_rng={"policy": "disabled"},
        qkv_layout="batch_heads_tokens_width",
        head_layout="separate_qkv_heads",
        scale_source="default_width",
        use_cache=False,
        output_attentions=False,
        rope_parameters={"enabled": False},
        position_id_policy={"type": "none"},
        score_softcap=0.75,
        final_logit_softcap=30.0,
    )
    settings = vpx.AttentionSettings(
        frontend="patched_eager",
        sdpa_kernel=None,
        sdpa_priority=(),
        partition="full",
        padding="dense_padded",
        query_block_size=None,
    )

    output = vpx.execute_attention(
        location,
        {"query": query, "key": key, "value": value},
        settings,
    )
    signature = location.signature()
    scores = query @ key.transpose(-2, -1)
    scores = scores / (float(query.size(-1)) ** 0.5)
    scores = torch.tanh(scores / 0.75) * 0.75
    expected = torch.softmax(scores, dim=-1) @ value

    assert isinstance(output, dict)
    assert len(output) == 1
    key, output_tensor = next(iter(output.items()))
    assert key == "attention"
    assert isinstance(output_tensor, torch.Tensor)
    torch.testing.assert_close(output_tensor, expected)
    assert signature["semantics"]["rope_parameters"] == {"enabled": False}
    assert signature["semantics"]["final_logit_softcap"] == pytest.approx(30.0)


def test_transformers_model_identity_changes_with_replay_inputs() -> None:
    model = TinyTiedModule()
    model.eval()
    base = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "sdpa"},
    )
    changed_config = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="other-config",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "sdpa"},
    )
    changed_tokenizer = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "other"},
        adapter_rules={"attention": "sdpa"},
    )
    changed_rules = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "eager"},
    )
    model.train()
    changed_module_state = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "sdpa"},
    )

    assert base.signature() != changed_config.signature()
    assert base.signature() != changed_tokenizer.signature()
    assert base.signature() != changed_rules.signature()
    assert base.signature() != changed_module_state.signature()


def test_transformers_model_identity_requires_explicit_fields() -> None:
    model = TinyTiedModule()

    with pytest.raises(AdmissionError, match="model_config_hash"):
        vpa.transformers_model_identity(
            model,
            transformers_version="4.0.0",
            model_config_hash="",
            source_revision=None,
            dtype_policy={"dtype.model_compute": "bf16"},
            tokenizer_identity={"name": "tokenizer"},
            adapter_rules={"attention": "sdpa"},
        )

    with pytest.raises(AdmissionError, match="tokenizer_identity"):
        vpa.transformers_model_identity(
            model,
            transformers_version="4.0.0",
            model_config_hash="config-hash",
            source_revision=None,
            dtype_policy={"dtype.model_compute": "bf16"},
            tokenizer_identity={},
            adapter_rules={"attention": "sdpa"},
        )


@pytest.mark.parametrize(
    "settings_override",
    [
        {"attention.partition": "full"},
        {"attention.padding": "dense_padded"},
        {"chunk.sequence_position_block_size": 2},
    ],
)
def test_transformers_attention_axis_rejects_core_attention_settings(
    settings_override: dict[str, object],
) -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(("transformers_sdpa",), policy=policy)
    settings = {
        "attention.frontend": "transformers_sdpa",
        "attention.sdpa_kernel": "math",
        "module_mode": "eval",
        "dropout_p": 0.0,
        **settings_override,
    }

    assert axis.admit(vpx.Candidate("family", "core-owned", settings)) == (
        False,
        f"{next(iter(settings_override))} is owned by the core attention executor",
    )


def test_transformers_sdpa_axis_rejects_unavailable_forced_kernel() -> None:
    policy = dataclasses.replace(
        transformers_policy(),
        forced_kernel_available=False,
        forced_kernel_failure_reason="kernel missing",
    )
    axis = vpa.transformers_attention_axis(("transformers_sdpa",), policy=policy)
    candidate = vpx.Candidate(
        "family",
        "efficient",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "efficient_attention",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )

    assert axis.admit(candidate) == (False, "kernel missing")


def test_transformers_attention_requires_mode_and_effective_flash_dtype() -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=policy,
    )
    missing_mode = vpx.Candidate(
        "family",
        "missing-mode",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "bf16",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    float32_compute = vpx.Candidate(
        "family",
        "float32-compute",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.parameter_storage": "bf16",
            "dtype.model_compute": "fp32",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    bf16_compute = vpx.Candidate(
        "family",
        "bf16-compute",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.parameter_storage": "fp32",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )

    assert axis.admit(missing_mode)[0] is False
    assert axis.admit(float32_compute)[0] is False
    assert axis.admit(bf16_compute) == (True, None)


def test_transformers_attention_axis_uses_core_and_adapter_admission_fields() -> None:
    policy = transformers_policy()
    registry = vpx.standard_axis_registry()
    registry.register(
        vpa.transformers_attention_axis(
            ("transformers_flash_attention_2", "registered_transformers_attention"),
            policy=policy,
        )
    )
    flash = vpx.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    registered = vpx.Candidate(
        "family",
        "registered",
        {
            "attention.frontend": "registered_transformers_attention",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "attention.custom_kernel_id": "custom_attention",
            "attention.mask_formatter_id": "custom_attention",
        },
    )
    mismatched_registered = vpx.Candidate(
        "family",
        "mismatched-registered",
        {
            "attention.frontend": "registered_transformers_attention",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "attention.custom_kernel_id": "custom_attention",
            "attention.mask_formatter_id": "custom_mask",
        },
    )
    missing_dropout = vpx.Candidate(
        "family",
        "missing-dropout",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
        },
    )

    assert registry.admit(flash).admission_status == "passed"
    assert registry.admit(registered).admission_status == "passed"
    assert registry.admit(mismatched_registered).admission_status == "failed"
    assert registry.admit(missing_dropout).admission_status == "failed"

    with pytest.raises(vp.AdmissionError, match="no axis owner"):
        registry.admit(
            vpx.Candidate(
                "family",
                "gqa-row-setting",
                {
                    "attention.frontend": "transformers_flash_attention_2",
                    "dtype.model_compute": "bf16",
                    "module_mode": "eval",
                    "dropout_p": 0.0,
                    "enable_gqa": True,
                    "query_heads": 4,
                    "key_heads": 2,
                    "value_heads": 2,
                },
            )
        )


def test_transformers_attention_axis_rejects_core_frontends() -> None:
    policy = transformers_policy()
    core_frontends = (
        "pytorch_sdpa_direct",
        "patched_eager",
        "packed_exact",
        "blockwise_exact",
    )

    with pytest.raises(vp.AdmissionError, match="unsupported Transformers attention"):
        vpa.transformers_attention_axis(core_frontends, policy=policy)
