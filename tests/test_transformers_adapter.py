import dataclasses
from collections.abc import Callable

import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
import vptune.ext as vpx
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


def fake_attention() -> None:
    return None


def fake_mask() -> None:
    return None


def test_load_transformers_model_calls_from_pretrained_with_explicit_settings() -> None:
    calls = []

    model = vpa.load_transformers_model(
        FakeModelLoader(calls),
        model_name_or_path="org/model",
        revision="abc123",
        torch_dtype=torch.bfloat16,
        attention_frontend="transformers_flash_attention_4",
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
    assert vpa.transformers_attn_implementation(attention_frontend) == expected


def test_registered_transformers_attn_implementation_uses_custom_id() -> None:
    assert (
        vpa.transformers_attn_implementation(
            "registered_transformers_attention",
            attention_custom_kernel_id="custom_attention",
        )
        == "custom_attention"
    )

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attn_implementation("pytorch_sdpa_direct")

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attn_implementation("registered_transformers_attention")


def test_set_transformers_attention_implementation_calls_model_method() -> None:
    model = FakeAttentionConfigurable()

    selected = vpa.set_transformers_attention_implementation(
        model,
        attention_frontend="transformers_sdpa",
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


def test_transformers_cache_axis_admission_and_identity() -> None:
    policy = transformers_policy(use_cache=False)
    axis = vpa.transformers_cache_axis(policy=policy)

    matching = vp.Candidate("family", "matching", {"use_cache": False})
    mismatched = vp.Candidate("family", "mismatched", {"use_cache": True})
    missing = vp.Candidate("family", "missing", {})

    assert axis.admit(matching) == (True, None)
    assert axis.admit(mismatched)[0] is False
    assert axis.admit(missing)[0] is False
    assert vpa.admit_transformers_cache(matching, policy=policy) == (True, None)
    assert vpa.admit_transformers_cache(mismatched, policy=policy)[0] is False

    signature = axis.signature()
    assert signature["adapter_id"] == "vptune.transformers"
    assert signature["adapter_version"] == PACKAGE_VERSION
    assert signature["settings_keys"] == ("use_cache",)
    assert signature["identity"]["use_cache"] is False


def test_transformers_cache_axis_composes_with_attention_axis() -> None:
    policy = transformers_policy(use_cache=True)
    registry = vpx.AxisRegistry()
    registry.register(
        vpa.transformers_attention_axis(
            ("transformers_sdpa",),
            policy=policy,
        )
    )
    registry.register(vpa.transformers_cache_axis(policy=policy))

    admitted = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention.frontend": "transformers_sdpa",
                "attention.sdpa_kernel": "math",
                "module_mode": "eval",
                "dropout_p": 0.0,
                "use_cache": True,
            },
        )
    )
    rejected = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention.frontend": "transformers_sdpa",
                "attention.sdpa_kernel": "math",
                "module_mode": "eval",
                "dropout_p": 0.0,
                "use_cache": False,
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"


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
