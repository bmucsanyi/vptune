import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
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
    registry = vp.AxisRegistry()
    registry.register(
        vpa.transformers_attention_axis(
            ("sdpa_math",),
            policy=policy,
        )
    )
    registry.register(vpa.transformers_cache_axis(policy=policy))

    admitted = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention_impl": "sdpa_math",
                "use_cache": True,
            },
        )
    )
    rejected = registry.admit(
        vp.Candidate(
            "family",
            "row",
            {
                "attention_impl": "sdpa_math",
                "use_cache": False,
            },
        )
    )

    assert admitted.admission_status == "passed"
    assert rejected.admission_status == "failed"


def test_transformers_attention_admits_packed_and_blockwise_rows() -> None:
    policy = transformers_policy()
    axis = vpa.transformers_attention_axis(
        ("packed_target_rows", "blockwise_second_derivative"),
        policy=policy,
    )
    packed = vp.Candidate(
        "family",
        "packed",
        {
            "attention_impl": "packed_target_rows",
            "packed_attention_id": "packed-gemma-target-rows",
            "packed_attention_semantics": {"mask": "causal", "softcap": 30.0},
            "packed_target_row_axis": "target_tokens",
            "packed_target_row_count": 4,
        },
    )
    missing_packed_semantics = vp.Candidate(
        "family",
        "missing-packed-semantics",
        {**packed.settings, "packed_attention_semantics": {}},
    )
    blockwise = vp.Candidate(
        "family",
        "blockwise",
        {
            "attention_impl": "blockwise_second_derivative",
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

    assert axis.admit(packed) == (True, None)
    assert axis.admit(missing_packed_semantics)[0] is False
    assert axis.admit(blockwise) == (True, None)
    assert axis.admit(invalid_block_size)[0] is False
    assert axis.admit(missing_blockwise_mask)[0] is False


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
