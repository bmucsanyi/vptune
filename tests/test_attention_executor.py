import contextlib
import dataclasses
import math
from collections.abc import Iterator
from typing import Any

import pytest
import torch

import vptune.attention as vpat
from vptune.errors import AdmissionError, ReferenceFailedError


def attention_inputs() -> vpat.AttentionInputs:
    query = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 17.0
    key = torch.arange(24, 48, dtype=torch.float32).reshape(1, 2, 4, 3) / 19.0
    value = torch.arange(48, 72, dtype=torch.float32).reshape(1, 2, 4, 3) / 23.0

    return vpat.AttentionInputs(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        score_softcap=None,
        enable_gqa=False,
        inverse_permutation=None,
        query_block_size=None,
    )


def attention_semantics(
    *,
    score_softcap: float | None = None,
    final_logit_softcap: float | None = None,
) -> vpat.AttentionSemantics:
    return vpat.AttentionSemantics(
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
        score_softcap=score_softcap,
        final_logit_softcap=final_logit_softcap,
    )


def attention_settings(
    frontend: str,
    sdpa_kernel: str | None,
    sdpa_priority: tuple[str, ...],
    partition: str,
    padding: str,
) -> vpat.AttentionSettings:
    return vpat.AttentionSettings(
        frontend=frontend,
        sdpa_kernel=sdpa_kernel,
        sdpa_priority=sdpa_priority,
        partition=partition,
        padding=padding,
    )


def test_pytorch_sdpa_direct_matches_exact_attention() -> None:
    inputs = attention_inputs()
    settings = attention_settings(
        "pytorch_sdpa_direct",
        "math",
        (),
        "full",
        "dense_padded",
    )

    direct = vpat.run_attention(inputs, settings)
    exact = vpat.exact_attention(inputs)

    torch.testing.assert_close(direct, exact)


def test_sdpa_priority_list_enters_priority_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    @contextlib.contextmanager
    def fake_sdpa_kernel(
        backends: list[Any] | Any,
        set_priority: bool,
    ) -> Iterator[None]:
        if isinstance(backends, list):
            calls.append((tuple(backends), set_priority))
        else:
            calls.append(((backends,), set_priority))

        yield

    monkeypatch.setattr(vpat, "sdpa_kernel", fake_sdpa_kernel)
    settings = attention_settings(
        "pytorch_sdpa_direct",
        "priority_list",
        ("flash_attention", "math"),
        "full",
        "dense_padded",
    )

    vpat.run_attention(attention_inputs(), settings)

    assert calls == [
        (
            (
                vpat.SDPA_BACKENDS["flash_attention"],
                vpat.SDPA_BACKENDS["math"],
            ),
            True,
        )
    ]


@pytest.mark.parametrize(
    "kernel_name",
    [
        "math",
        "flash_attention",
        "efficient_attention",
        "cudnn_attention",
        "overrideable",
    ],
)
def test_sdpa_kernel_values_enter_declared_context(
    monkeypatch: pytest.MonkeyPatch,
    kernel_name: str,
) -> None:
    calls = []

    @contextlib.contextmanager
    def fake_sdpa_kernel(
        backends: list[Any] | Any,
        set_priority: bool,
    ) -> Iterator[None]:
        if isinstance(backends, list):
            calls.append((tuple(backends), set_priority))
        else:
            calls.append(((backends,), set_priority))

        yield

    def fake_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: float | None,
        enable_gqa: bool,
    ) -> torch.Tensor:
        inputs = vpat.AttentionInputs(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            score_softcap=None,
            enable_gqa=enable_gqa,
            inverse_permutation=None,
            query_block_size=None,
        )

        return vpat.exact_attention(inputs)

    monkeypatch.setattr(vpat, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(
        vpat.functional,
        "scaled_dot_product_attention",
        fake_scaled_dot_product_attention,
    )
    settings = attention_settings(
        "pytorch_sdpa_direct",
        kernel_name,
        (),
        "full",
        "dense_padded",
    )
    output = vpat.run_attention(attention_inputs(), settings)

    torch.testing.assert_close(output, vpat.exact_attention(attention_inputs()))
    assert calls == [((vpat.SDPA_BACKENDS[kernel_name],), False)]


def test_mapping_attention_location_executes_non_transformers_attention() -> None:
    inputs = attention_inputs()
    location = vpat.MappingAttentionLocation(
        semantics=attention_semantics(),
        query_key="q",
        key_key="k",
        value_key="v",
        output_key="out",
        mask_key=None,
        inverse_permutation_key=None,
        query_block_size_key=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    )
    batch = {
        "q": inputs.query,
        "k": inputs.key,
        "v": inputs.value,
    }
    settings = attention_settings(
        "patched_eager",
        None,
        (),
        "full",
        "dense_padded",
    )

    output = attention_output_tensor(vpat.execute_attention(location, batch, settings))

    torch.testing.assert_close(output, vpat.exact_attention(inputs))
    assert location.signature()["query_key"] == "q"
    assert location.signature()["semantics"]["rope_parameters"] == {"enabled": False}


def test_packed_exact_attention_restores_token_order() -> None:
    inputs = attention_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse_permutation = torch.argsort(permutation)
    packed_inputs = dataclasses_replace_attention(
        inputs,
        query=inputs.query.index_select(-2, permutation),
        key=inputs.key.index_select(-2, permutation),
        value=inputs.value.index_select(-2, permutation),
        inverse_permutation=inverse_permutation,
    )
    settings = attention_settings(
        "packed_exact",
        None,
        (),
        "packed_tokens",
        "unpadded_packed",
    )

    output = vpat.run_attention(packed_inputs, settings)

    torch.testing.assert_close(output, vpat.exact_attention(inputs))


def test_blockwise_exact_attention_matches_full_attention() -> None:
    inputs = dataclasses_replace_attention(attention_inputs(), query_block_size=2)
    causal_inputs = dataclasses_replace_attention(inputs, is_causal=True)
    settings = attention_settings(
        "blockwise_exact",
        None,
        (),
        "blockwise_queries",
        "dense_padded",
    )

    output = vpat.run_attention(inputs, settings)
    causal_output = vpat.run_attention(causal_inputs, settings)

    torch.testing.assert_close(output, vpat.exact_attention(inputs))
    torch.testing.assert_close(causal_output, vpat.exact_attention(causal_inputs))


def test_exact_attention_applies_score_softcap() -> None:
    inputs = dataclasses_replace_attention(attention_inputs(), score_softcap=0.75)
    scores = inputs.query @ inputs.key.transpose(-2, -1)
    scores = scores / math.sqrt(inputs.query.size(-1))
    scores = torch.tanh(scores / 0.75) * 0.75
    expected = torch.softmax(scores, dim=-1) @ inputs.value

    output = vpat.exact_attention(inputs)

    torch.testing.assert_close(output, expected)


def test_apply_final_logit_softcap() -> None:
    logits = torch.tensor([[-3.0, -0.25, 0.0, 2.0]])
    expected = torch.tanh(logits / 1.5) * 1.5

    output = vpat.apply_final_logit_softcap(logits, 1.5)
    unchanged = vpat.apply_final_logit_softcap(logits, None)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(unchanged, logits)


def test_softcap_rejects_nonpositive_value() -> None:
    with pytest.raises(AdmissionError, match="final logit softcap must be positive"):
        vpat.apply_final_logit_softcap(torch.tensor([1.0]), 0.0)

    with pytest.raises(
        AdmissionError, match="attention score softcap must be positive"
    ):
        vpat.exact_attention(
            dataclasses_replace_attention(attention_inputs(), score_softcap=-1.0)
        )


def test_attention_executor_rejects_invalid_rows() -> None:
    inputs = attention_inputs()

    with pytest.raises(AdmissionError, match=r"requires attention[.]sdpa_kernel"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "pytorch_sdpa_direct",
                None,
                (),
                "full",
                "dense_padded",
            ),
        )

    with pytest.raises(AdmissionError, match="priority_list requires backend order"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "pytorch_sdpa_direct",
                "priority_list",
                (),
                "full",
                "dense_padded",
            ),
        )

    with pytest.raises(AdmissionError, match="partition=packed_tokens"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "packed_exact",
                None,
                (),
                "full",
                "unpadded_packed",
            ),
        )

    with pytest.raises(AdmissionError, match="requires query block size"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "blockwise_exact",
                None,
                (),
                "blockwise_queries",
                "dense_padded",
            ),
        )

    with pytest.raises(AdmissionError, match="applies only"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "patched_eager",
                "math",
                (),
                "full",
                "dense_padded",
            ),
        )

    with pytest.raises(AdmissionError, match=r"requires attention[.]partition=full"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "pytorch_sdpa_direct",
                "math",
                (),
                "blockwise_queries",
                "dense_padded",
            ),
        )

    with pytest.raises(AdmissionError, match=r"requires attention[.]padding"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "patched_eager",
                None,
                (),
                "full",
                "unpadded_packed",
            ),
        )

    with pytest.raises(AdmissionError, match="segmented_forward_ad"):
        vpat.run_attention(
            inputs,
            attention_settings(
                "blockwise_exact",
                None,
                (),
                "segmented_forward_ad",
                "dense_padded",
            ),
        )

    with pytest.raises(
        AdmissionError, match=r"blockwise_exact requires attention[.]padding"
    ):
        vpat.run_attention(
            inputs,
            attention_settings(
                "blockwise_exact",
                None,
                (),
                "blockwise_queries",
                "unpadded_packed",
            ),
        )

    with pytest.raises(AdmissionError, match="score softcap"):
        vpat.run_attention(
            dataclasses_replace_attention(inputs, score_softcap=0.75),
            attention_settings(
                "pytorch_sdpa_direct",
                "math",
                (),
                "full",
                "dense_padded",
            ),
        )


def test_patched_attention_output_reference_accepts_matching_outputs() -> None:
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

    measurements = vpat.check_patched_attention_output_reference(
        reference,
        patched,
        (query, key, value),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert measurements["max_abs_diff"] == pytest.approx(0.0)
    assert measurements["max_rel_diff"] == pytest.approx(0.0)


def test_patched_attention_output_reference_rejects_mismatched_outputs() -> None:
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
        vpat.check_patched_attention_output_reference(
            reference,
            patched,
            (query, key, value),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )


def test_patched_attention_vjp_reference_accepts_matching_derivatives() -> None:
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

    measurements = vpat.check_patched_attention_vjp_reference(
        reference,
        patched,
        (query, key, value),
        cotangent,
        (0, 1, 2),
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )

    assert measurements["max_abs_diff"] == pytest.approx(0.0)
    assert measurements["max_rel_diff"] == pytest.approx(0.0)


def test_patched_attention_vjp_reference_rejects_mismatched_derivatives() -> None:
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
        vpat.check_patched_attention_vjp_reference(
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
        vpat.check_patched_attention_vjp_reference(
            reference,
            reference,
            (query, key, value),
            cotangent,
            (),
            thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        )


def dataclasses_replace_attention(
    inputs: vpat.AttentionInputs,
    **changes: object,
) -> vpat.AttentionInputs:
    return dataclasses.replace(inputs, **changes)


def attention_output_tensor(tree: object) -> torch.Tensor:
    assert isinstance(tree, dict)
    assert set(tree) == {"out"}
    key, output = next(iter(tree.items()))
    assert key == "out"
    assert isinstance(output, torch.Tensor)

    return output
