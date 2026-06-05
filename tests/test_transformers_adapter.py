import contextlib
import dataclasses
from collections.abc import Callable, Iterator

import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
import vptune.adapters.transformers as transformers_module
import vptune.ext as vpx
import vptune.runtime as runtime_module
from vptune.data import PACKAGE_VERSION
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


class FakeModelLoader:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def from_pretrained(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        use_cache: bool,
    ) -> torch.nn.Module:
        self.calls.append({
            "model_name_or_path": model_name_or_path,
            "revision": revision,
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "use_cache": use_cache,
        })

        return TinyTiedModule()


class FakeTransformersRegistry:
    def __init__(self) -> None:
        self.calls = []

    def register(self, name: str, function: Callable[..., object]) -> None:
        self.calls.append((name, function))


class FakeAttentionConfigurable:
    def __init__(self) -> None:
        self.values = []

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.values.append(attn_implementation)


class TinyTransformersScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.attention_values = []

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.attention_values.append(attn_implementation)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.w * scale).sum()


class ContextCheckingTransformersModule(TinyTransformersScalarModule):
    def __init__(self, active: list[bool], forward_events: list[bool]) -> None:
        super().__init__()
        self.active = active
        self.forward_events = forward_events

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        self.forward_events.append(bool(self.active))

        return super().forward(scale)


def fake_attention() -> None:
    return None


def fake_mask() -> None:
    return None


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


def test_load_transformers_model_calls_from_pretrained_with_explicit_settings() -> None:
    calls = []

    model = vpa.load_transformers_model(
        FakeModelLoader(calls),
        model_name_or_path="org/model",
        revision="abc123",
        torch_dtype=torch.bfloat16,
        attention_frontend="transformers_flash_attention_4",
        attention_custom_kernel_id=None,
        use_cache=False,
    )

    assert isinstance(model, TinyTiedModule)
    assert calls == [
        {
            "model_name_or_path": "org/model",
            "revision": "abc123",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_4",
            "use_cache": False,
        }
    ]


def test_transformers_patched_attention_output_reference_returns_result() -> None:
    query = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    key = torch.tensor([[0.25, -0.5], [1.5, 0.75]])
    value = torch.tensor([[0.0, 1.0], [2.0, -3.0]])

    def reference(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    result = vpa.check_patched_attention_output_reference(
        reference,
        reference,
        (query, key, value),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert result.name == "patched_attention_output"
    assert result.thresholds == {"max_abs_diff": 0.0, "max_rel_diff": 0.0}
    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_transformers_patched_attention_vjp_reference_returns_result() -> None:
    query = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    key = torch.tensor([[0.25, -0.5], [1.5, 0.75]])
    value = torch.tensor([[0.0, 1.0], [2.0, -3.0]])
    cotangent = torch.ones(2, 2)

    def reference(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    result = vpa.check_patched_attention_vjp_reference(
        reference,
        reference,
        (query, key, value),
        cotangent,
        (0, 1, 2),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert result.name == "patched_attention_vjp"
    assert result.thresholds == {"max_abs_diff": 0.0, "max_rel_diff": 0.0}
    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_transformers_operation_factory_sets_attention_and_runs_module() -> None:
    model = TinyTransformersScalarModule()
    factory = vpa.transformers_operation_factory(
        vp.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    output = factory(
        vp.Candidate(
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
    candidate = vp.Candidate(
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
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    monkeypatch.setattr(transformers_module, "sdpa_kernel", fake_sdpa_kernel)
    factory = vpa.transformers_operation_factory(
        vp.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    check = vpa.transformers_reference_check(
        vp.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
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
    candidate = vp.Candidate(
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
        operation: Callable[[], vp.TensorTree],
        **_: object,
    ) -> Callable[[], vp.TensorTree]:
        return operation

    monkeypatch.setattr(transformers_module, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(runtime_module.torch, "compile", fake_compile)
    factory = vpa.transformers_operation_factory(
        vp.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
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


def test_transformers_reference_check_runs_independent_anchor() -> None:
    model = TinyTransformersScalarModule()

    def scalar_objective(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    check = vpa.transformers_reference_check(
        vp.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar_objective},
    )
    result = check(
        vp.Candidate(
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
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    candidate = vp.Candidate(
        "gradient",
        "transformers-gradient",
        stateful_transformers_settings(),
        admission_status="passed",
    )
    runtime = vpa.transformers_runtime_config(
        vp.gradient("gradient", "loss", aggregation="sum"),
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
        module_call=vp.ModuleCallSpec(positional_batch_keys=("scale",)),
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
    record = vp.FullSizeRecord(
        family="gradient",
        candidate_id="transformers-gradient",
        status="passed",
        input_signature={},
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    selected = runtime.materializer(candidate, record)
    selected_output = selected(batch, vector)
    selected_map = tensor_dict(selected_output)

    assert runtime.identity()["full_size_check"] is not None
    assert metadata["transformers_full_size_checked_inputs"] == 1
    torch.testing.assert_close(
        selected_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_load_transformers_model_forwards_registered_attention_id() -> None:
    calls = []

    vpa.load_transformers_model(
        FakeModelLoader(calls),
        model_name_or_path="org/model",
        revision="abc123",
        torch_dtype=torch.float16,
        attention_frontend="registered_transformers_attention",
        attention_custom_kernel_id="custom_attention",
        use_cache=True,
    )

    assert calls == [
        {
            "model_name_or_path": "org/model",
            "revision": "abc123",
            "torch_dtype": torch.float16,
            "attn_implementation": "custom_attention",
            "use_cache": True,
        }
    ]


@pytest.mark.parametrize(
    ("attention_frontend", "expected"),
    [
        ("transformers_eager", "eager"),
        ("transformers_sdpa", "sdpa"),
        ("transformers_flash_attention_2", "flash_attention_2"),
        ("transformers_flash_attention_3", "flash_attention_3"),
        ("transformers_flash_attention_4", "flash_attention_4"),
        ("transformers_flex_attention", "flex_attention"),
        ("paged|eager", "paged|eager"),
        ("paged|sdpa", "paged|sdpa"),
        ("paged|flash_attention_2", "paged|flash_attention_2"),
        ("paged|flash_attention_3", "paged|flash_attention_3"),
        ("paged|flash_attention_4", "paged|flash_attention_4"),
    ],
)
def test_transformers_attn_implementation_maps_load_time_frontends(
    attention_frontend: str,
    expected: str,
) -> None:
    assert vpa.transformers_attn_implementation(attention_frontend, None) == expected


def test_registered_transformers_attn_implementation_uses_custom_id() -> None:
    assert (
        vpa.transformers_attn_implementation(
            "registered_transformers_attention",
            "custom_attention",
        )
        == "custom_attention"
    )

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attn_implementation("pytorch_sdpa_direct", None)

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attn_implementation("registered_transformers_attention", None)


def test_set_transformers_attention_implementation_calls_model_method() -> None:
    model = FakeAttentionConfigurable()

    selected = vpa.set_transformers_attention_implementation(
        model,
        attention_frontend="transformers_sdpa",
        attention_custom_kernel_id=None,
    )
    custom = vpa.set_transformers_attention_implementation(
        model,
        attention_frontend="registered_transformers_attention",
        attention_custom_kernel_id="custom_attention",
    )

    assert selected == "sdpa"
    assert custom == "custom_attention"
    assert model.values == ["sdpa", "custom_attention"]


def test_register_transformers_attention_registers_attention_and_mask() -> None:
    attention_registry = FakeTransformersRegistry()
    mask_registry = FakeTransformersRegistry()

    identity = vpa.register_transformers_attention(
        attention_registry,
        mask_registry,
        attention_custom_kernel_id="custom_attention",
        attention_function=fake_attention,
        mask_formatter_id="custom_attention",
        mask_function=fake_mask,
    )

    assert attention_registry.calls == [("custom_attention", fake_attention)]
    assert mask_registry.calls == [("custom_attention", fake_mask)]
    assert identity["attention_custom_kernel_id"] == "custom_attention"
    assert identity["mask_formatter_id"] == "custom_attention"

    with pytest.raises(vp.AdmissionError, match="matching attention and mask"):
        vpa.register_transformers_attention(
            attention_registry,
            mask_registry,
            attention_custom_kernel_id="custom_attention",
            attention_function=fake_attention,
            mask_formatter_id="custom_mask",
            mask_function=fake_mask,
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


def test_transformers_model_identity_records_module_and_adapter_fields() -> None:
    model = TinyTiedModule()
    model.eval()

    identity = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"dtype.model_compute": "bf16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "sdpa"},
    )

    signature = identity.signature()

    assert signature["adapter_id"] == "vptune.transformers"
    assert signature["adapter_version"] == PACKAGE_VERSION
    assert signature["transformers_version"] == "4.0.0"
    assert signature["model_config_hash"] == "config-hash"
    assert signature["source_revision"] == "revision"
    assert signature["dtype_policy"] == {"dtype.model_compute": "bf16"}
    assert signature["tokenizer_identity"] == {
        "name": "tokenizer",
        "revision": "tok-rev",
    }
    assert signature["adapter_rules"] == {"attention": "sdpa"}
    assert signature["module"]["training"] is False
    assert signature["module"]["tied_parameter_groups"] == (("embedding", "lm_head"),)
    assert signature["module"]["buffers"][0]["name"] == "scale"


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
    "sdpa_kernel",
    [
        "math",
        "flash_attention",
        "efficient_attention",
        "cudnn_attention",
        "overrideable",
    ],
)
def test_transformers_sdpa_axis_admits_each_kernel_value(sdpa_kernel: str) -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(("transformers_sdpa",), policy=policy)
    settings = {
        "attention.frontend": "transformers_sdpa",
        "attention.sdpa_kernel": sdpa_kernel,
        "module_mode": "eval",
        "dropout_p": 0.0,
    }

    if sdpa_kernel == "flash_attention":
        settings["dtype.model_compute"] = "bf16"

    assert axis.admit(vp.Candidate("family", sdpa_kernel, settings)) == (True, None)


def test_transformers_sdpa_axis_admits_priority_list() -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(("transformers_sdpa",), policy=policy)
    candidate = vp.Candidate(
        "family",
        "priority",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )

    assert axis.admit(candidate) == (True, None)


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

    assert axis.admit(vp.Candidate("family", "core-owned", settings)) == (
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
    candidate = vp.Candidate(
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


def test_transformers_attention_registers_adapter_axis() -> None:
    policy = transformers_policy()
    registry = vpx.standard_axis_registry()
    registry.register(
        vpa.transformers_attention_axis(
            ("transformers_eager",),
            policy=policy,
        )
    )
    admitted = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention.frontend": "transformers_eager",
                "module_mode": "eval",
                "dropout_p": 0.0,
            },
        )
    )

    assert admitted.admission_status == "passed"


def test_transformers_attention_requires_mode_and_effective_flash_dtype() -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=policy,
    )
    missing_mode = vp.Candidate(
        "family",
        "missing-mode",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "bf16",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    float32_compute = vp.Candidate(
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
    bf16_compute = vp.Candidate(
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


def test_transformers_attention_axis_owns_optional_admission_fields() -> None:
    policy = transformers_policy()
    registry = vpx.AxisRegistry()
    registry.register(
        vpa.transformers_attention_axis(
            ("transformers_flash_attention_2", "registered_transformers_attention"),
            policy=policy,
        )
    )
    flash = vp.Candidate(
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
    registered = vp.Candidate(
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
    mismatched_registered = vp.Candidate(
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
    missing_dropout = vp.Candidate(
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
