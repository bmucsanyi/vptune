import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
import torch

import vptune as vp
import vptune.attention as vpat
import vptune.ext as vpx
import vptune.runtime as runtime_module
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
    query_block_size: int | None = None,
) -> vpat.AttentionSettings:
    return vpat.AttentionSettings(
        frontend=frontend,
        sdpa_kernel=sdpa_kernel,
        sdpa_priority=sdpa_priority,
        partition=partition,
        padding=padding,
        query_block_size=query_block_size,
    )


def attention_location() -> vpat.MappingAttentionLocation:
    return vpat.MappingAttentionLocation(
        semantics=attention_semantics(),
        query_key="query",
        key_key="key",
        value_key="value",
        output_key="out",
        mask_key=None,
        inverse_permutation_key=None,
        query_block_size_key="query_block_size",
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    )


def test_core_attention_axis_admits_executable_frontends() -> None:
    registry = vpx.AxisRegistry()
    registry.register(vpat.core_attention_axis())
    rows = (
        vp.Candidate(
            "attention",
            "sdpa",
            {
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
                "attention.partition": "full",
                "attention.padding": "dense_padded",
            },
        ),
        vp.Candidate(
            "attention",
            "patched",
            {
                "attention.frontend": "patched_eager",
                "attention.partition": "full",
                "attention.padding": "dense_padded",
            },
        ),
        vp.Candidate(
            "attention",
            "packed",
            {
                "attention.frontend": "packed_exact",
                "attention.partition": "packed_tokens",
                "attention.padding": "unpadded_packed",
            },
        ),
        vp.Candidate(
            "attention",
            "blockwise",
            {
                "attention.frontend": "blockwise_exact",
                "attention.partition": "blockwise_queries",
                "attention.padding": "dense_padded",
                "chunk.sequence_position_block_size": 2,
            },
        ),
        vp.Candidate(
            "attention",
            "segmented-forward-ad",
            {
                "attention.frontend": "blockwise_exact",
                "attention.partition": "segmented_forward_ad",
                "attention.padding": "dense_padded",
            },
        ),
    )

    for row in rows:
        assert registry.admit(row).admission_status == "passed"


def test_core_attention_axis_rejects_invalid_sequence_block_size() -> None:
    registry = vpx.AxisRegistry()
    registry.register(vpat.core_attention_axis())
    row = vp.Candidate(
        "attention",
        "bad-block",
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
            "attention.padding": "dense_padded",
            "chunk.sequence_position_block_size": 0,
        },
    )

    assert registry.admit(row).admission_status == "failed"


@pytest.mark.parametrize(
    "settings",
    [
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "unknown",
            "attention.partition": "full",
            "attention.padding": "dense_padded",
        },
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "math",
            "attention.padding": "dense_padded",
        },
        {
            "attention.frontend": "packed_exact",
            "attention.partition": "segmented_forward_ad",
            "attention.padding": "unpadded_packed",
        },
    ],
)
def test_core_attention_axis_rejects_invalid_rows(
    settings: dict[str, object],
) -> None:
    registry = vpx.AxisRegistry()
    registry.register(vpat.core_attention_axis())

    admitted = registry.admit(vp.Candidate("attention", "row", settings))

    assert admitted.admission_status == "failed"


def test_attention_settings_from_candidate_records_priority_order() -> None:
    settings = vpat.attention_settings_from_candidate({
        "attention.frontend": "pytorch_sdpa_direct",
        "attention.sdpa_kernel": "priority_list",
        "attention.sdpa_priority_list": ("flash_attention", "math"),
        "attention.partition": "full",
        "attention.padding": "dense_padded",
    })

    assert settings.signature() == {
        "frontend": "pytorch_sdpa_direct",
        "sdpa_kernel": "priority_list",
        "sdpa_priority": ("flash_attention", "math"),
        "partition": "full",
        "padding": "dense_padded",
        "query_block_size": None,
    }


def test_attention_settings_from_candidate_records_sequence_block_size() -> None:
    settings = vpat.attention_settings_from_candidate({
        "attention.frontend": "blockwise_exact",
        "attention.partition": "blockwise_queries",
        "attention.padding": "dense_padded",
        "chunk.sequence_position_block_size": 2,
    })

    assert settings.signature()["query_block_size"] == 2


def test_attention_operation_factory_and_reference_check_execute_core_row() -> None:
    inputs = attention_inputs()
    batch = {
        "query": inputs.query,
        "key": inputs.key,
        "value": inputs.value,
        "query_block_size": 2,
    }
    candidate = vp.Candidate(
        "attention",
        "blockwise",
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
            "attention.padding": "dense_padded",
        },
    )
    location = attention_location()
    factory = vpat.attention_operation_factory(location)
    reference_check = vpat.attention_reference_check(
        location,
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
    )
    output = factory(candidate, batch, {})()
    result = reference_check(candidate, batch, {})
    expected = vpat.exact_attention(inputs)

    torch.testing.assert_close(attention_output_tensor(output), expected)
    assert result.name == "core_attention_reference"
    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_attention_reference_check_rejects_dropout_without_reproducible_policy() -> (
    None
):
    inputs = attention_inputs()
    batch = {
        "query": inputs.query,
        "key": inputs.key,
        "value": inputs.value,
    }
    location = vpat.MappingAttentionLocation(
        semantics=attention_semantics(),
        query_key="query",
        key_key="key",
        value_key="value",
        output_key="out",
        mask_key=None,
        inverse_permutation_key=None,
        query_block_size_key=None,
        dropout_p=0.25,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    )
    reference_check = vpat.attention_reference_check(
        location,
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
    )

    with pytest.raises(AdmissionError, match=r"dropout_p=0[.]0"):
        reference_check(
            vp.Candidate(
                "attention",
                "dropout",
                {
                    "attention.frontend": "patched_eager",
                    "attention.partition": "full",
                    "attention.padding": "dense_padded",
                },
            ),
            batch,
            {},
        )


def test_attention_operation_uses_candidate_sequence_block_size() -> None:
    inputs = attention_inputs()
    batch = {
        "query": inputs.query,
        "key": inputs.key,
        "value": inputs.value,
    }
    candidate = vp.Candidate(
        "attention",
        "blockwise",
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
            "attention.padding": "dense_padded",
            "chunk.sequence_position_block_size": 2,
        },
    )
    location = vpat.MappingAttentionLocation(
        semantics=attention_semantics(),
        query_key="query",
        key_key="key",
        value_key="value",
        output_key="out",
        mask_key=None,
        inverse_permutation_key=None,
        query_block_size_key=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    )
    factory = vpat.attention_operation_factory(location)
    output = factory(candidate, batch, {})()
    expected = vpat.exact_attention(inputs)

    torch.testing.assert_close(attention_output_tensor(output), expected)


def test_attention_operation_compiles_attention_module_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    def fake_compile(
        operation: Callable[[], vp.TensorTree],
        *,
        backend: str,
        mode: str | None,
        fullgraph: bool,
        dynamic: bool | None,
        options: Mapping[str, bool] | None,
    ) -> Callable[[], vp.TensorTree]:
        events.append({
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "options": options,
        })

        def compiled() -> vp.TensorTree:
            events.append({"compiled_attention": True})

            return operation()

        return compiled

    monkeypatch.setattr(runtime_module.torch, "compile", fake_compile)
    location = attention_location()
    batch = {
        "query": attention_inputs().query,
        "key": attention_inputs().key,
        "value": attention_inputs().value,
        "query_block_size": 2,
    }
    factory = vpat.attention_operation_factory(location)
    operation = factory(
        vp.Candidate(
            "attention",
            "compiled-attention",
            {
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
                "attention.partition": "full",
                "attention.padding": "dense_padded",
                "compile.enabled": "true",
                "compile.boundary": "attention_module",
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
        batch,
        {},
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
    output = attention_output_tensor(operation())
    expected = vpat.exact_attention(attention_inputs())

    assert events[-1] == {"compiled_attention": True}
    torch.testing.assert_close(output, expected)


def test_attention_operation_rejects_other_compile_boundaries() -> None:
    factory = vpat.attention_operation_factory(attention_location())

    with pytest.raises(vp.MaterializationError, match="model_forward"):
        factory(
            vp.Candidate(
                "attention",
                "bad-compile",
                {
                    "attention.frontend": "pytorch_sdpa_direct",
                    "attention.sdpa_kernel": "math",
                    "attention.partition": "full",
                    "attention.padding": "dense_padded",
                    "compile.enabled": "true",
                    "compile.boundary": "model_forward",
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
            {
                "query": attention_inputs().query,
                "key": attention_inputs().key,
                "value": attention_inputs().value,
            },
            {},
        )


def test_attention_rejects_conflicting_query_block_sizes() -> None:
    inputs = attention_inputs()
    batch = {
        "query": inputs.query,
        "key": inputs.key,
        "value": inputs.value,
        "query_block_size": 3,
    }
    candidate = vp.Candidate(
        "attention",
        "blockwise",
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
            "attention.padding": "dense_padded",
            "chunk.sequence_position_block_size": 2,
        },
    )
    factory = vpat.attention_operation_factory(attention_location())

    with pytest.raises(vp.AdmissionError, match="query block sizes differ"):
        factory(candidate, batch, {})()


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


def test_flash_sdpa_matches_math_backend_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Flash SDPA")

    major, _ = torch.cuda.get_device_capability()

    if major < 8:
        pytest.skip("Flash SDPA requires Ampere or newer CUDA hardware")

    torch.manual_seed(0)
    query = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
    value = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.float16)
    inputs = vpat.AttentionInputs(
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
    flash = vpat.run_attention(
        inputs,
        attention_settings(
            "pytorch_sdpa_direct",
            "flash_attention",
            (),
            "full",
            "dense_padded",
        ),
    )
    math_output = vpat.run_attention(
        inputs,
        attention_settings(
            "pytorch_sdpa_direct",
            "math",
            (),
            "full",
            "dense_padded",
        ),
    )

    torch.testing.assert_close(
        flash.float(),
        math_output.float(),
        atol=3e-2,
        rtol=3e-2,
    )


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


def test_packed_exact_attention_restores_padded_positions() -> None:
    inputs = attention_inputs()
    packed_indices = torch.tensor([2, 0, 3])
    inverse_permutation = torch.tensor([1, -1, 0, 2, -1])
    packed_inputs = dataclasses_replace_attention(
        inputs,
        query=inputs.query.index_select(-2, packed_indices),
        key=inputs.key.index_select(-2, packed_indices),
        value=inputs.value.index_select(-2, packed_indices),
        inverse_permutation=inverse_permutation,
    )
    settings = attention_settings(
        "packed_exact",
        None,
        (),
        "packed_tokens",
        "unpadded_packed",
    )
    packed_output = vpat.exact_attention(packed_inputs)
    expected = packed_output.new_zeros(
        *packed_output.shape[:-2],
        5,
        packed_output.size(-1),
    )
    expected.index_copy_(
        -2,
        torch.tensor([0, 2, 3]),
        packed_output.index_select(-2, torch.tensor([1, 0, 2])),
    )

    output = vpat.run_attention(packed_inputs, settings)

    torch.testing.assert_close(output, expected)


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


def test_segmented_forward_ad_attention_matches_full_attention_tangent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = dataclasses_replace_attention(attention_inputs(), query_block_size=2)
    tangent = torch.linspace(
        0.1,
        0.4,
        steps=inputs.query.numel(),
        dtype=inputs.query.dtype,
    ).reshape_as(inputs.query)
    settings = attention_settings(
        "blockwise_exact",
        None,
        (),
        "segmented_forward_ad",
        "dense_padded",
    )

    def blocked_block_attention(
        inputs: vpat.AttentionInputs,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        assert inputs
        assert stop >= start
        message = "segmented_forward_ad used blockwise query helper"
        raise AssertionError(message)

    monkeypatch.setattr(vpat, "_block_attention", blocked_block_attention)

    with torch.autograd.forward_ad.dual_level():
        dual_query = torch.autograd.forward_ad.make_dual(inputs.query, tangent)
        dual_inputs = dataclasses_replace_attention(inputs, query=dual_query)
        full = vpat.exact_attention(dual_inputs)
        segmented = vpat.run_attention(dual_inputs, settings)
        full_primal, full_tangent = torch.autograd.forward_ad.unpack_dual(full)
        segmented_primal, segmented_tangent = torch.autograd.forward_ad.unpack_dual(
            segmented
        )

    torch.testing.assert_close(segmented_primal, full_primal)
    torch.testing.assert_close(segmented_tangent, full_tangent)


def test_segmented_forward_ad_attention_matches_causal_tangent() -> None:
    inputs = dataclasses_replace_attention(
        attention_inputs(),
        query_block_size=2,
        is_causal=True,
    )
    tangent = torch.linspace(
        0.1,
        0.4,
        steps=inputs.query.numel(),
        dtype=inputs.query.dtype,
    ).reshape_as(inputs.query)
    settings = attention_settings(
        "blockwise_exact",
        None,
        (),
        "segmented_forward_ad",
        "dense_padded",
    )

    with torch.autograd.forward_ad.dual_level():
        dual_query = torch.autograd.forward_ad.make_dual(inputs.query, tangent)
        dual_inputs = dataclasses_replace_attention(inputs, query=dual_query)
        full = vpat.exact_attention(dual_inputs)
        segmented = vpat.run_attention(dual_inputs, settings)
        full_primal, full_tangent = torch.autograd.forward_ad.unpack_dual(full)
        segmented_primal, segmented_tangent = torch.autograd.forward_ad.unpack_dual(
            segmented
        )

    torch.testing.assert_close(segmented_primal, full_primal)
    torch.testing.assert_close(segmented_tangent, full_tangent)


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

    with pytest.raises(AdmissionError, match="blockwise_exact requires"):
        vpat.run_attention(
            dataclasses_replace_attention(inputs, query_block_size=2),
            attention_settings(
                "blockwise_exact",
                None,
                (),
                "full",
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
