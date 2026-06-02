import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
import vptune.ext as vpx
from vptune.data import PACKAGE_VERSION
from vptune.errors import AdmissionError, ReferenceFailedError


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


def test_transformers_attn_implementation_maps_load_time_frontends() -> None:
    assert vpa.transformers_attn_implementation("transformers_sdpa") == "sdpa"
    assert (
        vpa.transformers_attn_implementation("paged|flash_attention_3")
        == "paged|flash_attention_3"
    )
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


def test_transformers_model_identity_records_module_and_adapter_fields() -> None:
    model = TinyTiedModule()
    model.eval()

    identity = vpa.transformers_model_identity(
        model,
        transformers_version="4.0.0",
        model_config_hash="config-hash",
        source_revision="revision",
        dtype_policy={"model_dtype": "bfloat16"},
        tokenizer_identity={"name": "tokenizer", "revision": "tok-rev"},
        adapter_rules={"attention": "sdpa"},
    )

    signature = identity.signature()

    assert signature["adapter_id"] == "vptune.transformers"
    assert signature["adapter_version"] == PACKAGE_VERSION
    assert signature["transformers_version"] == "4.0.0"
    assert signature["model_config_hash"] == "config-hash"
    assert signature["source_revision"] == "revision"
    assert signature["dtype_policy"] == {"model_dtype": "bfloat16"}
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
            dtype_policy={"model_dtype": "bfloat16"},
            tokenizer_identity={"name": "tokenizer"},
            adapter_rules={"attention": "sdpa"},
        )

    with pytest.raises(AdmissionError, match="tokenizer_identity"):
        vpa.transformers_model_identity(
            model,
            transformers_version="4.0.0",
            model_config_hash="config-hash",
            source_revision=None,
            dtype_policy={"model_dtype": "bfloat16"},
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
            ("pytorch_sdpa_direct",),
            policy=policy,
        )
    )
    registry.register(vpa.transformers_cache_axis(policy=policy))

    admitted = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention.frontend": "pytorch_sdpa_direct",
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
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
                "module_mode": "eval",
                "dropout_p": 0.0,
                "use_cache": False,
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"


def test_transformers_attention_replaces_core_attention_axis() -> None:
    policy = transformers_policy()
    registry = vpx.standard_axis_registry(
        exclude=("attention_frontend", "functional_call_admission")
    )
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
            "model_dtype": "bfloat16",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    float32_compute = vp.Candidate(
        "family",
        "float32-compute",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "model_dtype": "bfloat16",
            "compute_dtype": "float32",
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
            "model_dtype": "float32",
            "compute_dtype": "bfloat16",
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
            ("transformers_flash_attention_2", "blockwise_exact"),
            policy=policy,
        )
    )
    flash = vp.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": False,
        },
    )
    blockwise = vp.Candidate(
        "family",
        "blockwise",
        {
            "attention.frontend": "blockwise_exact",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "blockwise_attention_id": "blockwise-gemma-hvp",
            "blockwise_attention_semantics": {"mask": "causal", "softcap": 30.0},
            "attention_block_size": 16,
            "blockwise_preserves_softcap": True,
            "blockwise_preserves_mask": True,
        },
    )
    missing_dropout = vp.Candidate(
        "family",
        "missing-dropout",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "model_dtype": "bfloat16",
            "module_mode": "eval",
        },
    )

    assert registry.admit(flash).admission_status == "passed"
    assert registry.admit(blockwise).admission_status == "passed"
    assert registry.admit(missing_dropout).admission_status == "failed"


def test_transformers_attention_admits_packed_and_blockwise_rows() -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(
        ("packed_exact", "blockwise_exact"),
        policy=policy,
    )
    packed = vp.Candidate(
        "family",
        "packed",
        {
            "attention.frontend": "packed_exact",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "packed_attention_id": "packed-gemma-target-rows",
            "packed_attention_semantics": {"mask": "causal", "softcap": 30.0},
            "packed_target_row_axis": "target_tokens",
            "packed_target_row_count": 4,
            "packed_preserves_softcap": True,
            "packed_preserves_mask": True,
            "packed_mask_semantics": "boolean_keep_mask",
            "packed_causal_policy": "causal",
        },
    )
    missing_packed_semantics = vp.Candidate(
        "family",
        "missing-packed-semantics",
        {**packed.settings, "packed_attention_semantics": {}},
    )
    missing_packed_mask = vp.Candidate(
        "family",
        "missing-packed-mask",
        {**packed.settings, "packed_preserves_mask": None},
    )
    false_packed_softcap = vp.Candidate(
        "family",
        "false-packed-softcap",
        {**packed.settings, "packed_preserves_softcap": False},
    )
    wrong_packed_causal_policy = vp.Candidate(
        "family",
        "wrong-packed-causal-policy",
        {**packed.settings, "packed_causal_policy": "bidirectional"},
    )
    blockwise = vp.Candidate(
        "family",
        "blockwise",
        {
            "attention.frontend": "blockwise_exact",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "blockwise_attention_id": "blockwise-gemma-hvp",
            "blockwise_attention_semantics": {"mask": "causal", "softcap": 30.0},
            "attention_block_size": 16,
            "blockwise_preserves_softcap": True,
            "blockwise_preserves_mask": True,
        },
    )
    invalid_block_size = vp.Candidate(
        "family",
        "invalid-block-size",
        {**blockwise.settings, "attention_block_size": 0},
    )
    missing_blockwise_mask = vp.Candidate(
        "family",
        "missing-blockwise-mask",
        {**blockwise.settings, "blockwise_preserves_mask": None},
    )
    false_softcap = vp.Candidate(
        "family",
        "false-softcap",
        {**blockwise.settings, "blockwise_preserves_softcap": False},
    )
    false_mask = vp.Candidate(
        "family",
        "false-mask",
        {**blockwise.settings, "blockwise_preserves_mask": False},
    )

    assert axis.admit(packed) == (True, None)
    assert axis.admit(missing_packed_semantics)[0] is False
    assert axis.admit(missing_packed_mask)[0] is False
    assert axis.admit(false_packed_softcap)[0] is False
    assert axis.admit(wrong_packed_causal_policy)[0] is False
    assert axis.admit(blockwise) == (True, None)
    assert axis.admit(invalid_block_size)[0] is False
    assert axis.admit(missing_blockwise_mask)[0] is False
    assert axis.admit(false_softcap)[0] is False
    assert axis.admit(false_mask)[0] is False


def test_patched_attention_reference_check_accepts_matching_outputs() -> None:
    query = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    key = torch.tensor([[0.25, -0.5], [1.5, 0.75]])
    value = torch.tensor([[0.0, 1.0], [2.0, -3.0]])

    def reference(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    def patched(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    measurements = vpa.check_patched_attention_reference(
        reference,
        patched,
        (query, key, value),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert measurements["max_abs_diff"] == pytest.approx(0.0)
    assert measurements["max_rel_diff"] == pytest.approx(0.0)


def test_patched_attention_reference_check_rejects_mismatched_outputs() -> None:
    query = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    key = torch.tensor([[0.25, -0.5], [1.5, 0.75]])
    value = torch.tensor([[0.0, 1.0], [2.0, -3.0]])

    def reference(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    def patched(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value + 0.5

    with pytest.raises(ReferenceFailedError, match="measurement exceeds threshold"):
        vpa.check_patched_attention_reference(
            reference,
            patched,
            (query, key, value),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )


def test_patched_attention_vjp_reference_check_accepts_matching_derivatives() -> None:
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

    def patched(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return input_query @ input_key.T + input_value

    measurements = vpa.check_patched_attention_vjp_reference(
        reference,
        patched,
        (query, key, value),
        cotangent,
        (0, 1, 2),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert measurements["max_abs_diff"] == pytest.approx(0.0)
    assert measurements["max_rel_diff"] == pytest.approx(0.0)


def test_patched_attention_vjp_reference_check_rejects_mismatched_derivatives() -> None:
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

    def patched(
        input_query: torch.Tensor,
        input_key: torch.Tensor,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        return 1.1 * (input_query @ input_key.T) + input_value

    with pytest.raises(ReferenceFailedError, match="measurement exceeds threshold"):
        vpa.check_patched_attention_vjp_reference(
            reference,
            patched,
            (query, key, value),
            cotangent,
            (0, 1, 2),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )


def test_patched_attention_vjp_reference_requires_explicit_arg_indices() -> None:
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

    with pytest.raises(AdmissionError, match="non-empty"):
        vpa.check_patched_attention_vjp_reference(
            reference,
            reference,
            (query, key, value),
            cotangent,
            (),
            thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        )
