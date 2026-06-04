"""Candidate axes, admission, and DAG helpers."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import Any

from vptune.admission import (
    FORWARD_AD_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_forward_ad,
    admit_torch_func,
)
from vptune.checks import (
    NUMERIC_ERROR_BOUND_FIELDS,
    uses_reduction_degrading_setting,
)
from vptune.data import Candidate, Family
from vptune.errors import AdmissionError

AdmissionRule = Callable[[Candidate], tuple[bool, str | None]]


@dataclasses.dataclass(frozen=True, slots=True)
class AxisTableDescriptor:
    """One sweep axis entry from the fixed docs."""

    axis_key: str
    owner_id: str
    value_domain: tuple[Any, ...]
    operators: tuple[str, ...]
    class_c_group: str
    admission_rule_id: str
    lowering_rule_id: str
    class_a: str = ""
    class_b: str = ""
    adapter_id: str = ""
    merge_rules: tuple[str, ...] = ()

    def signature(self) -> dict[str, Any]:
        """Return stable axis identity.

        Returns:
            Serializable axis identity.
        """
        return {
            "axis_key": self.axis_key,
            "owner_id": self.owner_id,
            "value_domain": self.value_domain,
            "operators": self.operators,
            "class_a": self.class_a,
            "class_b": self.class_b,
            "class_c_group": self.class_c_group,
            "merge_rules": self.merge_rules,
            "admission_rule_id": self.admission_rule_id,
            "lowering_rule_id": self.lowering_rule_id,
            "adapter_id": self.adapter_id,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class AxisTable:
    """Complete sweep axis table."""

    package_version: str
    axis_table_version: str
    axes: tuple[AxisTableDescriptor, ...]
    class_c_groups: Mapping[str, tuple[str, ...]]
    merge_rules: tuple[str, ...]

    def by_key(self) -> dict[str, AxisTableDescriptor]:
        """Return axes keyed by setting key.

        Returns:
            Mapping from axis key to descriptor.
        """
        return {axis.axis_key: axis for axis in self.axes}

    def signature(self) -> dict[str, Any]:
        """Return stable axis table identity.

        Returns:
            Serializable axis table identity.
        """
        return {
            "package_version": self.package_version,
            "axis_table_version": self.axis_table_version,
            "axes": tuple(axis.signature() for axis in self.axes),
            "class_c_groups": dict(sorted(self.class_c_groups.items())),
            "merge_rules": self.merge_rules,
        }

    def admit(
        self,
        candidate: Candidate,
        *,
        fixed_fields: Mapping[str, Any],
    ) -> Candidate:
        """Return candidate with axis table admission status set.

        Returns:
            Candidate with updated admission status.
        """
        error = self.admission_error(candidate, fixed_fields=fixed_fields)

        if error is not None:
            return dataclasses.replace(
                candidate,
                admission_status="failed",
                admission_error=error,
            )

        return dataclasses.replace(candidate, admission_status="passed")

    def admission_error(
        self,
        candidate: Candidate,
        *,
        fixed_fields: Mapping[str, Any],
    ) -> str | None:
        """Return the first axis table admission error.

        Returns:
            Failure reason, or None when admitted.
        """
        by_key = self.by_key()

        for key, value in candidate.settings.items():
            axis = by_key.get(key)

            if axis is None:
                return f"candidate setting key has no axis table owner: {key}"

            value_error = _axis_table_value_error(axis, value)

            if value_error is not None:
                return value_error

        return _axis_table_cross_rule_error(candidate.settings, fixed_fields)


@dataclasses.dataclass(frozen=True, slots=True)
class AxisTableAdmitter:
    """Candidate admitter backed by the axis table."""

    axis_table: AxisTable
    fixed_fields: Mapping[str, Any]

    def admit(self, candidate: Candidate) -> Candidate:
        """Return candidate with admission status set."""
        return self.axis_table.admit(candidate, fixed_fields=self.fixed_fields)

    def signature(self) -> dict[str, Any]:
        """Return stable admission identity."""
        return {
            "axis_table": self.axis_table.signature(),
            "fixed_fields": dict(self.fixed_fields),
        }


ALL_OPERATOR_FAMILIES = (
    "gradient",
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "metric",
    "inverse_metric",
    "composition",
)
MODEL_CALL_OPERATOR_FAMILIES = (
    "gradient",
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
)
VECTOR_OPERATOR_FAMILIES = (
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "composition",
)

PACKED_ATTENTION_MERGE_RULE = (
    "attention.partition=packed_tokens merges attention_dispatch with input_schedule"
)
ATTENTION_COMPILE_MERGE_RULE = (
    "compile.boundary=attention_module merges compile with attention_dispatch"
)
OPERATOR_COMPILE_MERGE_RULE = (
    "compile.boundary operator part merges compile with ad_lowering"
)
FUSION_MERGE_RULE = "fusion non-default merges fusion with ad_lowering"
DTENSOR_MERGE_RULE = "dtensor placement merges distributed_layout with ad_lowering"
FSDP_REDUCE_DTYPE_MERGE_RULE = (
    "fsdp reduce dtype not fp32 merges distributed_layout with numeric_backend"
)
FACTORIZED_INVERSE_MERGE_RULE = (
    "factorized inverse rows merge inverse_solve with metric_storage"
)

AXIS_TABLE_MERGE_RULES = (
    PACKED_ATTENTION_MERGE_RULE,
    ATTENTION_COMPILE_MERGE_RULE,
    OPERATOR_COMPILE_MERGE_RULE,
    FUSION_MERGE_RULE,
    DTENSOR_MERGE_RULE,
    FSDP_REDUCE_DTYPE_MERGE_RULE,
    FACTORIZED_INVERSE_MERGE_RULE,
)

INTEGER_DOMAIN = ("positive_integer_domain",)
INTEGER_TUPLE_DOMAIN = ("positive_integer_tuple_domain",)
POSITIVE_FLOAT_DOMAIN = ("positive_float_domain",)
DECLARED_DOMAIN = ("declared",)
REGISTERED_DOMAIN = ("registered",)
BACKEND_OPTION_KEYS = (
    "compile.options.epilogue_fusion",
    "compile.options.shape_padding",
    "compile.cuda_graphs",
)
COMPILE_DISABLED_VALUES = {
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
PACKED_BATCH_LAYOUTS = ("packed_with_inverse_permutation", "variable_length")
CHECKPOINT_RECOMPUTE_VALUES = (
    "checkpoint_non_reentrant_by_layer",
    "checkpoint_selective",
)
CHECKPOINT_DISABLED_SETTINGS = {
    "checkpoint.early_stop": "false",
    "checkpoint.preserve_rng_state": "false",
    "checkpoint.determinism_check": "none",
    "checkpoint.context_fn": "none",
}
FORWARD_AD_PATHS_BY_KEY = {
    "jvp.path": ("torch_func_jvp", "forward_ad_dual"),
    "hvp.path": ("jvp_grad", "forward_ad_dual"),
    "ggn.jvp_path": ("torch_func_jvp", "forward_ad_dual"),
}
MICROBATCH_OPERATOR_PATH_KEYS = (
    "gradient.path",
    "jvp.path",
    "vjp.path",
    "hvp.path",
    "ggn.jvp_path",
    "fisher.accumulation",
    "sampled_fisher.accumulation",
    "empirical_fisher.grad_path",
)
DENSE_REPRESENTATIONS = ("dense_matrix",)
DIAGONAL_REPRESENTATIONS = ("diagonal_tree",)
BLOCK_REPRESENTATIONS = ("block_diagonal",)
KFAC_REPRESENTATIONS = ("kfac_factors",)
LOW_RANK_REPRESENTATIONS = ("low_rank_factors",)
GGN_REPRESENTATIONS = ("ggn_derived_factors",)
FACTOR_REPRESENTATIONS = (
    *DIAGONAL_REPRESENTATIONS,
    *KFAC_REPRESENTATIONS,
    *LOW_RANK_REPRESENTATIONS,
    *GGN_REPRESENTATIONS,
)
STREAMING_REPRESENTATIONS = (
    *DIAGONAL_REPRESENTATIONS,
    *BLOCK_REPRESENTATIONS,
    *KFAC_REPRESENTATIONS,
    *LOW_RANK_REPRESENTATIONS,
    *GGN_REPRESENTATIONS,
)
BLOCK_OR_KFAC_REPRESENTATIONS = (*BLOCK_REPRESENTATIONS, *KFAC_REPRESENTATIONS)
DIRECT_SOLVE_PATHS = ("dense_solve", "cholesky_solve", "eigh_solve", "svd_solve")
ITERATIVE_SOLVE_PATHS = ("conjugate_gradient",)
ADMISSION_BOUND_FIELDS = NUMERIC_ERROR_BOUND_FIELDS
OWNER_EXACT = {
    "teacher_outputs": "input_schedule",
    "autocast": "numeric_backend",
}
OWNER_PREFIX = {
    "gradient": "gradient",
    "jvp": "jvp",
    "vjp": "vjp",
    "hvp": "hvp",
    "ggn": "ggn",
    "fisher": "fisher",
    "sampled_fisher": "sampled_fisher",
    "empirical_fisher": "empirical_fisher",
    "metric": "metric",
    "inverse_metric": "inverse_metric",
    "composition": "composition",
    "vectorization": "vectorization",
    "compile": "compile",
    "fusion": "fusion",
    "call": "model_call",
    "attention": "attention_execution",
    "batch": "input_schedule",
    "chunk": "input_schedule",
    "schedule": "input_schedule",
    "input": "input_schedule",
    "dtype": "numeric_backend",
    "numeric": "numeric_backend",
    "layout": "layout",
    "distributed": "distributed",
    "dtensor": "distributed",
    "fsdp": "distributed",
    "tp": "distributed",
    "sequence_parallel": "distributed",
    "context_parallel": "distributed",
    "comm": "distributed",
}
CLASS_C_EXACT = {
    "teacher_outputs": "input_schedule",
    "autocast": "numeric_backend",
    "memory.primal_outputs": "activation_memory",
    "memory.jvp_outputs": "activation_memory",
    "memory.output_cotangents": "activation_memory",
    "memory.vector_residency": "activation_memory",
    "memory.intermediate_residency": "activation_memory",
    "memory.factor_residency": "metric_storage",
    "memory.output_buffers": "compile",
}
CLASS_C_PREFIX = {
    "gradient": "ad_lowering",
    "jvp": "ad_lowering",
    "vjp": "ad_lowering",
    "hvp": "ad_lowering",
    "ggn": "ad_lowering",
    "fisher": "ad_lowering",
    "sampled_fisher": "ad_lowering",
    "empirical_fisher": "ad_lowering",
    "composition": "ad_lowering",
    "vectorization": "ad_lowering",
    "call": "ad_lowering",
    "attention": "attention_dispatch",
    "batch": "input_schedule",
    "chunk": "input_schedule",
    "schedule": "input_schedule",
    "input": "input_schedule",
    "checkpoint": "activation_memory",
    "activation": "activation_memory",
    "dtype": "numeric_backend",
    "numeric": "numeric_backend",
    "layout": "distributed_layout",
    "dtensor": "distributed_layout",
    "distributed": "distributed_layout",
    "fsdp": "distributed_layout",
    "tp": "distributed_layout",
    "sequence_parallel": "distributed_layout",
    "context_parallel": "distributed_layout",
    "comm": "distributed_layout",
    "compile": "compile",
    "fusion": "fusion",
    "metric": "metric_storage",
    "inverse_metric": "inverse_solve",
}
OPERATOR_PREFIX = {
    "gradient": ("gradient",),
    "jvp": ("jvp",),
    "vjp": ("vjp",),
    "hvp": ("hvp",),
    "ggn": ("ggnvp",),
    "fisher": ("fisher_vp",),
    "sampled_fisher": ("sampled_fisher_vp",),
    "empirical_fisher": ("empirical_fisher_vp",),
    "metric": ("metric",),
    "inverse_metric": ("inverse_metric",),
    "composition": ("composition",),
}


def axis_table() -> AxisTable:
    """Return the complete sweep axis table.

    Returns:
        Complete axis table for candidate generation and admission.
    """
    axes = tuple(starmap(_axis_table_axis, _axis_table_axis_domains()))
    groups = {}

    for axis in axes:
        groups.setdefault(axis.class_c_group, []).append(axis.axis_key)

    class_c_groups = {name: tuple(keys) for name, keys in sorted(groups.items())}

    return AxisTable(
        package_version="0.0.1",
        axis_table_version="1",
        axes=axes,
        class_c_groups=class_c_groups,
        merge_rules=AXIS_TABLE_MERGE_RULES,
    )


def _axis_table_axis(
    axis_key: str,
    value_domain: tuple[Any, ...],
) -> AxisTableDescriptor:
    class_c_group = _class_c_group(axis_key)

    return AxisTableDescriptor(
        axis_key=axis_key,
        owner_id=_owner_id(axis_key),
        value_domain=value_domain,
        operators=_operators_for_axis(axis_key),
        class_a=_class_a(axis_key),
        class_b=_class_b(axis_key),
        class_c_group=class_c_group,
        merge_rules=_axis_merge_rules(axis_key),
        admission_rule_id=f"admit.{axis_key}",
        lowering_rule_id=f"lower.{axis_key}",
        adapter_id=_adapter_id(axis_key),
    )


def _axis_table_axis_domains() -> tuple[tuple[str, tuple[Any, ...]], ...]:
    return (
        (
            "gradient.path",
            (
                "torch_autograd_grad",
                "torch_func_grad",
                "torch_func_grad_and_value",
                "backward_materialized_grad",
            ),
        ),
        ("gradient.value_reuse", ("gradient_only", "gradient_and_primal_value")),
        ("gradient.graph_schedule", ("build_once", "rebuild_per_call")),
        ("jvp.path", ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize")),
        ("jvp.linearize_reuse", ("none", "reuse_at_same_primal")),
        (
            "vjp.path",
            ("torch_func_vjp", "autograd_grad_outputs", "backward_materialized_grad"),
        ),
        ("vjp.closure_reuse", ("none", "reuse_vjp_closure_at_same_primal")),
        (
            "hvp.path",
            (
                "reverse_over_reverse",
                "jvp_grad",
                "autograd_functional_hvp",
                "autograd_functional_vhp",
                "forward_ad_dual",
                "linearize_grad",
            ),
        ),
        (
            "hvp.graph_schedule",
            ("retain_graph_across_vectors", "rebuild_graph_per_vector"),
        ),
        ("hvp.primal_reuse", ("reuse_primal", "recompute_primal")),
        (
            "hvp.gradient_reuse",
            ("reuse_gradient_closure", "recompute_gradient"),
        ),
        (
            "ggn.jvp_path",
            ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize"),
        ),
        ("ggn.loss_hessian_path", ("closed_form_softmax_ce_kl", "autodiff_loss_hvp")),
        (
            "ggn.loss_hessian_kernel",
            ("dense_global", "streaming_global", "two_pass_chunked_global"),
        ),
        ("ggn.vjp_path", ("torch_func_vjp", "autograd_grad_outputs")),
        ("ggn.jvp_reuse", ("reuse_jvp", "recompute_jvp")),
        (
            "ggn.cotangent_reuse",
            ("reuse_output_cotangent", "recompute_output_cotangent"),
        ),
        ("fisher.expectation_path", ("explicit_full_expectation_score_rows",)),
        (
            "fisher.score_grad_path",
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
        ),
        (
            "fisher.accumulation",
            (
                "streaming_dot_accumulate",
                "materialize_score_gradients",
                "blockwise_score_matrix",
            ),
        ),
        (
            "sampled_fisher.sample_source",
            ("fixed_sample_table", "fixed_seed_and_count"),
        ),
        (
            "sampled_fisher.score_grad_path",
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
        ),
        (
            "sampled_fisher.accumulation",
            (
                "streaming_dot_accumulate",
                "materialize_score_gradients",
                "blockwise_score_matrix",
            ),
        ),
        (
            "sampled_fisher.exact_fisher_check",
            ("disabled", "enabled_with_sampling_bound"),
        ),
        (
            "empirical_fisher.grad_path",
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
        ),
        (
            "empirical_fisher.accumulation",
            (
                "streaming_dot_accumulate",
                "materialize_per_example_gradients",
                "blockwise_gradient_matrix",
            ),
        ),
        (
            "metric.multiply_path",
            (
                "dense_matmul",
                "factorized_multiply",
                "blockwise_multiply",
                "streaming_multiply",
            ),
        ),
        ("metric.block_schedule", ("layer_blocks", "module_blocks", "custom_blocks")),
        ("metric.accumulation", ("streaming", "materialized_blocks")),
        (
            "inverse_metric.solve_path",
            (
                "dense_solve",
                "cholesky_solve",
                "eigh_solve",
                "svd_solve",
                "conjugate_gradient",
                "factorized_solve",
                "blockwise_solve",
                "woodbury_low_rank_solve",
            ),
        ),
        (
            "inverse_metric.preconditioner",
            ("none", "diagonal", "block_diagonal", "factorized_metric"),
        ),
        ("inverse_metric.iteration_budget", INTEGER_DOMAIN),
        (
            "inverse_metric.factor_reuse",
            ("refactor_each_rhs", "reuse_factor_across_rhs"),
        ),
        (
            "inverse_metric.block_schedule",
            ("layer_blocks", "module_blocks", "custom_blocks"),
        ),
        (
            "composition.execution",
            (
                "materialize_each_child",
                "stream_child_outputs",
                "fuse_adjacent_children",
                "compile_whole_composition",
            ),
        ),
        (
            "composition.child_evaluation",
            ("selected_child_rows", "inline_child_lowering"),
        ),
        ("composition.validation", ("validate_each_child", "validate_composed_output")),
        ("vectorization.mode", ("single_loop", "manual_batch", "vmap")),
        ("vectorization.batch_size", INTEGER_DOMAIN),
        ("vectorization.vmap_chunk_size", INTEGER_DOMAIN),
        ("vectorization.in_dims", DECLARED_DOMAIN),
        ("vectorization.randomness", ("error", "same", "different")),
        ("call.path", ("functional_call", "stateful_module")),
        ("call.params", ("explicit_params", "module_params")),
        ("call.buffers", ("explicit_buffers", "module_buffers")),
        ("call.tied_weights", ("preserve_alias_groups",)),
        ("call.parametrizations", ("preserve_parametrizations",)),
        ("call.buffer_mutation", ("forbidden", "declared_and_restored")),
        ("call.grad_mode", ("grad_enabled",)),
        (
            "call.return_type",
            ("raw_tensor_tree", "model_output_object_with_declared_fields"),
        ),
        ("attention.frontend", ATTENTION_FRONTEND_VALUES),
        ("attention.sdpa_kernel", SDPA_KERNEL_VALUES),
        ("attention.custom_kernel_id", REGISTERED_DOMAIN),
        ("attention.mask_formatter_id", REGISTERED_DOMAIN),
        (
            "attention.partition",
            ("full", "packed_tokens", "blockwise_queries", "segmented_forward_ad"),
        ),
        ("attention.padding", ("dense_padded", "unpadded_packed")),
        ("batch.data_microbatch_size", INTEGER_DOMAIN),
        ("batch.hvp_row_batch_size", INTEGER_DOMAIN),
        ("batch.ggn_batch_size", INTEGER_DOMAIN),
        ("batch.fisher_sample_batch_size", INTEGER_DOMAIN),
        ("batch.empirical_example_batch_size", INTEGER_DOMAIN),
        ("chunk.token_block_size", INTEGER_DOMAIN),
        ("chunk.sequence_position_block_size", INTEGER_DOMAIN),
        ("chunk.class_block_size_with_exact_global_normalization", INTEGER_DOMAIN),
        ("chunk.output_cotangent_block_size", INTEGER_DOMAIN),
        ("chunk.parameter_block_size", INTEGER_DOMAIN),
        ("chunk.layer_block_size", INTEGER_DOMAIN),
        ("chunk.lm_head_weight_chunk_bytes", INTEGER_DOMAIN),
        ("schedule.per_example", ("loop", "vmap", "manual_batch")),
        ("schedule.per_token", ("loop", "packed")),
        ("schedule.gradient_accumulation", ("single_step", "microbatch_accumulate")),
        (
            "input.batch_layout",
            ("dense_padded", "packed_with_inverse_permutation", "variable_length"),
        ),
        ("input.length_grouping", ("none", "exact_length_bucket")),
        ("input.host_to_device", ("outside_measured_call", "inside_measured_call")),
        ("input.residency", ("cpu_staged", "cpu_pinned", "gpu")),
        (
            "teacher_outputs",
            (
                "precomputed_cpu",
                "precomputed_cpu_pinned",
                "precomputed_gpu",
                "recomputed_with_equality_check",
            ),
        ),
        ("checkpoint.use_reentrant", ("false",)),
        ("checkpoint.early_stop", ("false", "true")),
        ("checkpoint.preserve_rng_state", ("false", "true")),
        ("checkpoint.determinism_check", ("default", "none")),
        ("checkpoint.context_fn", ("none", "declared_context_pair")),
        ("memory.primal_outputs", ("retain", "recompute")),
        ("memory.jvp_outputs", ("retain", "recompute")),
        ("memory.output_cotangents", ("retain", "recompute")),
        (
            "activation.recompute",
            (
                "none",
                "checkpoint_non_reentrant_by_layer",
                "checkpoint_selective",
                "manual_recompute",
            ),
        ),
        (
            "activation.offload",
            ("none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"),
        ),
        ("memory.vector_residency", ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu")),
        ("memory.intermediate_residency", ("gpu", "cpu_pinned", "cpu_staged")),
        ("memory.factor_residency", ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu")),
        ("memory.output_buffers", ("fresh_allocation", "preallocated")),
        ("dtype.parameter_storage", ("fp32", "bf16", "fp16", "fp8_when_supported")),
        ("dtype.model_compute", ("fp32", "bf16", "fp16", "fp8_when_supported")),
        ("dtype.autodiff_compute", ("fp32", "bf16", "fp16")),
        ("dtype.accumulation", ("fp32", "bf16", "fp16")),
        ("dtype.vector", ("fp32", "bf16", "fp16")),
        ("dtype.intermediate", ("fp32", "bf16", "fp16")),
        ("dtype.output", ("fp32", "bf16", "fp16")),
        ("dtype.metric_factor", ("fp32", "bf16", "fp16")),
        ("autocast", ("off", "cuda_fp16", "cuda_bf16")),
        ("numeric.float32_matmul_precision", MATMUL_PRECISION_VALUES),
        ("numeric.bf16_reduced_precision_reduction", ("false", "true")),
        ("numeric.fp16_reduced_precision_reduction", ("false", "true")),
        ("numeric.deterministic_algorithms", ("false", "true")),
        ("numeric.loss_scaling", ("none", "static_scale_with_exact_unscale")),
        ("numeric.loss_scale", POSITIVE_FLOAT_DOMAIN),
        ("numeric.loss_unscale_degree", (1, 2)),
        ("compile.enabled", ("false", "true")),
        (
            "compile.boundary",
            (
                "model_forward",
                "transformer_block",
                "attention_module",
                "loss_closure",
                "gradient_closure",
                "jvp_closure",
                "vjp_closure",
                "hvp_single_vector",
                "hvp_batched_vectors",
                "ggn_jvp",
                "ggn_loss_hessian_product",
                "ggn_vjp",
                "ggn_full_product",
                "fisher_score_grad",
                "sampled_fisher_score_grad",
                "empirical_fisher_example_grad",
                "metric_multiply",
                "inverse_metric_solve",
                "composition_child",
                "whole_operator",
            ),
        ),
        ("compile.backend", ("inductor", "registered_backend")),
        ("compile.mode", (None, "default", "max-autotune")),
        ("compile.fullgraph", ("false", "true")),
        ("compile.dynamic", (None, "false", "true")),
        ("compile.compiled_autograd", ("false", "true")),
        ("compile.options.epilogue_fusion", ("false", "true")),
        ("compile.options.shape_padding", ("false", "true")),
        ("compile.cuda_graphs", ("false", "true")),
        ("compile.cache_state", ("cold_compile", "warm_cache")),
        ("fusion.norm", ("model_default", "fused_rmsnorm", "fused_layernorm")),
        ("fusion.mlp", ("model_default", "fused_mlp")),
        ("fusion.rope", ("model_default", "fused_rope")),
        ("fusion.logits", ("model_default", "fused_logits_projection")),
        ("fusion.loss", ("model_default", "fused_ce", "fused_kl")),
        (
            "layout.params",
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        (
            "layout.vector",
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        (
            "layout.output",
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        ("layout.flatten_order", ("canonical_parameter_order",)),
        ("layout.vector_ops", ("python_loop", "foreach")),
        ("layout.contiguity", ("contiguous", "preserve_existing_strides")),
        ("layout.aliasing", ("preserve_tied_weight_aliases",)),
        ("layout.parametrizations", ("preserve_active_parametrizations",)),
        ("distributed.launch", ("single_process", "torchrun")),
        ("distributed.process_group_backend", ("nccl", "gloo", "ucc_when_available")),
        ("distributed.local_rank_binding", ("cuda_local_rank", "explicit_device_map")),
        ("distributed.mesh_shape", INTEGER_TUPLE_DOMAIN),
        ("distributed.mesh_dim_names", DECLARED_DOMAIN),
        (
            "distributed.strategy",
            (
                "single_gpu",
                "fsdp2",
                "hsdp",
                "tensor_parallel",
                "sequence_parallel",
                "context_parallel",
                "hybrid",
            ),
        ),
        ("dtensor.params_placement", ("replicate", "shard_dim", "partial")),
        ("dtensor.vector_placement", ("replicate", "shard_dim", "partial")),
        ("dtensor.logits_placement", ("replicate", "shard_dim", "partial")),
        ("dtensor.tangent_placement", ("replicate", "shard_dim", "partial")),
        ("dtensor.cotangent_placement", ("replicate", "shard_dim", "partial")),
        ("dtensor.output_placement", ("replicate", "shard_dim", "partial")),
        (
            "dtensor.redistribute_schedule",
            (
                "none",
                "before_forward",
                "before_backward",
                "between_operator_parts",
                "before_output",
            ),
        ),
        ("fsdp.wrap_granularity", ("root", "transformer_block", "block_group")),
        (
            "fsdp.reshard_after_forward",
            ("false", "true", "positive_integer_group_size"),
        ),
        ("fsdp.shard_placement_fn", ("none", "declared_fn")),
        ("fsdp.mp_policy.param_dtype", ("fp32", "bf16", "fp16")),
        ("fsdp.mp_policy.reduce_dtype", ("fp32", "bf16", "fp16")),
        ("fsdp.mp_policy.output_dtype", ("fp32", "bf16", "fp16")),
        ("fsdp.mp_policy.cast_forward_inputs", ("false", "true")),
        ("fsdp.offload_policy", ("none", "cpu")),
        ("fsdp.ignored_params", DECLARED_DOMAIN),
        ("fsdp.dp_mesh_dims", DECLARED_DOMAIN),
        ("tp.plan", REGISTERED_DOMAIN),
        ("tp.qkv_projection", ("colwise", "rowwise", "replicated")),
        ("tp.output_projection", ("rowwise", "colwise", "replicated")),
        ("tp.mlp_up_gate", ("colwise", "rowwise", "replicated")),
        ("tp.mlp_down", ("rowwise", "colwise", "replicated")),
        ("tp.embedding", ("replicated", "rowwise", "colwise")),
        ("tp.lm_head", ("replicated", "vocab_sharded")),
        ("tp.prepare_module_input", DECLARED_DOMAIN),
        ("tp.prepare_module_output", DECLARED_DOMAIN),
        ("tp.loss_parallel", ("false", "true")),
        ("sequence_parallel.enabled", ("false", "true")),
        ("sequence_parallel.norm_modules", DECLARED_DOMAIN),
        (
            "sequence_parallel.output_placement_policy",
            ("preserve_sequence_shard", "redistribute_to_declared_output"),
        ),
        ("context_parallel.enabled", ("false", "true")),
        ("context_parallel.rotate_method", ("all_gather", "all_to_all")),
        ("context_parallel.sequence_dim", DECLARED_DOMAIN),
        (
            "comm.overlap",
            ("none", "all_gather_overlap", "reduce_scatter_overlap", "both"),
        ),
        ("comm.prefetch", ("none", "forward", "backward", "both")),
        ("comm.collective_bucket_size", INTEGER_DOMAIN),
    )


def _owner_id(axis_key: str) -> str:
    prefix = axis_key.split(".", 1)[0]
    exact_owner = OWNER_EXACT.get(axis_key)

    if exact_owner is not None:
        return exact_owner

    if prefix in {"checkpoint", "activation", "memory"}:
        return _memory_owner_id(axis_key)

    prefix_owner = OWNER_PREFIX.get(prefix)

    if prefix_owner is not None:
        return prefix_owner

    message = f"axis table axis has no owner: {axis_key}"
    raise AdmissionError(message)


def _memory_owner_id(axis_key: str) -> str:
    if axis_key == "memory.factor_residency":
        return "metric_storage"

    if axis_key == "memory.output_buffers":
        return "compile"

    return "activation_memory"


def _class_c_group(axis_key: str) -> str:
    prefix = axis_key.split(".", 1)[0]
    exact_group = CLASS_C_EXACT.get(axis_key)

    if exact_group is not None:
        return exact_group

    prefix_group = CLASS_C_PREFIX.get(prefix)

    if prefix_group is not None:
        return prefix_group

    message = f"axis table axis has no Class C group: {axis_key}"
    raise AdmissionError(message)


def _operators_for_axis(axis_key: str) -> tuple[str, ...]:
    prefix = axis_key.split(".", 1)[0]
    operators = OPERATOR_PREFIX.get(prefix)

    if operators is not None:
        return operators

    if prefix == "vectorization":
        return VECTOR_OPERATOR_FAMILIES

    if prefix in {"call", "attention"}:
        return MODEL_CALL_OPERATOR_FAMILIES

    return ALL_OPERATOR_FAMILIES


def _class_a(axis_key: str) -> str:
    if axis_key in {"layout.vector_ops", "gradient.value_reuse"}:
        return "mostly_independent_after_admission"

    return ""


def _class_b(axis_key: str) -> str:
    if axis_key in {
        "input.residency",
        "input.host_to_device",
        "compile.cache_state",
        "teacher_outputs",
    }:
        return "conditionally_independent"

    return ""


def _axis_merge_rules(axis_key: str) -> tuple[str, ...]:
    rules = []

    if axis_key == "attention.partition":
        rules.append(PACKED_ATTENTION_MERGE_RULE)

    if axis_key == "compile.boundary":
        rules.extend((ATTENTION_COMPILE_MERGE_RULE, OPERATOR_COMPILE_MERGE_RULE))

    if axis_key.startswith("fusion."):
        rules.append(FUSION_MERGE_RULE)

    if axis_key.startswith("dtensor."):
        rules.append(DTENSOR_MERGE_RULE)

    if axis_key == "fsdp.mp_policy.reduce_dtype":
        rules.append(FSDP_REDUCE_DTYPE_MERGE_RULE)

    if axis_key in {"inverse_metric.solve_path", "inverse_metric.preconditioner"}:
        rules.append(FACTORIZED_INVERSE_MERGE_RULE)

    return tuple(rules)


def _adapter_id(axis_key: str) -> str:
    prefix = axis_key.split(".", 1)[0]

    if prefix in {
        "distributed",
        "dtensor",
        "fsdp",
        "tp",
        "sequence_parallel",
        "context_parallel",
        "comm",
    }:
        return "vptune.adapters.distributed"

    if axis_key == "attention.frontend":
        return "vptune.attention_or_adapter"

    return ""


def _axis_table_value_error(
    axis: AxisTableDescriptor,
    value: Any,
) -> str | None:
    validators = (
        (INTEGER_DOMAIN, _positive_integer_value_error),
        (INTEGER_TUPLE_DOMAIN, _positive_integer_tuple_value_error),
        (POSITIVE_FLOAT_DOMAIN, _positive_float_value_error),
        (DECLARED_DOMAIN, _declared_value_error),
        (REGISTERED_DOMAIN, _registered_value_error),
    )

    for domain, validate in validators:
        if axis.value_domain == domain:
            return validate(axis.axis_key, value)

    if any(value == allowed for allowed in axis.value_domain):
        return None

    return f"candidate axis value is not allowed: {axis.axis_key}"


def _positive_integer_value_error(axis_key: str, value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return f"candidate axis must be a positive integer: {axis_key}"

    return None


def _positive_integer_tuple_value_error(axis_key: str, value: Any) -> str | None:
    if not isinstance(value, tuple) or not value:
        return f"candidate axis must be a positive integer tuple: {axis_key}"

    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return f"candidate axis must be a positive integer tuple: {axis_key}"

    return None


def _positive_float_value_error(axis_key: str, value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        return f"candidate axis must be a positive float: {axis_key}"

    return None


def _declared_value_error(axis_key: str, value: Any) -> str | None:
    if value is None or (isinstance(value, str) and len(value) == 0):
        return f"candidate axis requires a declared value: {axis_key}"

    return None


def _registered_value_error(axis_key: str, value: Any) -> str | None:
    if not isinstance(value, str) or len(value) == 0:
        return f"candidate axis requires a registered id: {axis_key}"

    return None


def _axis_table_cross_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    checks = (
        _sdpa_rule_error,
        _packing_rule_error,
        _activation_rule_error,
        _compile_rule_error,
        _reduction_bound_rule_error,
        _sampled_fisher_rule_error,
        _gradient_value_reuse_rule_error,
        _jvp_linearize_reuse_rule_error,
        _vjp_closure_reuse_rule_error,
        _ggn_vjp_path_rule_error,
        _fisher_rule_error,
        _iteration_budget_rule_error,
        _segmented_forward_ad_rule_error,
        _metric_representation_rule_error,
        _loss_scaling_rule_error,
        _tp_loss_parallel_rule_error,
    )

    for check in checks:
        error = check(settings, fixed_fields)

        if error is not None:
            return error

    return None


def _sdpa_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    kernel = settings.get("attention.sdpa_kernel")

    if kernel is None:
        return None

    if fixed_fields.get("attention.calls_sdpa") is not True:
        return "attention.sdpa_kernel requires an executable that calls PyTorch SDPA"

    if kernel == "priority_list":
        priority = fixed_fields.get("attention.sdpa_priority")

        if not isinstance(priority, tuple) or not priority:
            return "attention.sdpa_kernel=priority_list requires backend order"

    if kernel == "flash_attention":
        dtype = fixed_fields.get("attention.effective_runtime_dtype")

        if dtype not in {"float16", "bfloat16"}:
            return "flash attention requires float16 or bfloat16 runtime dtype"

    return None


def _packing_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    if (
        settings.get("schedule.per_token") == "packed"
        or settings.get("attention.partition") == "packed_tokens"
    ) and settings.get("input.batch_layout") not in PACKED_BATCH_LAYOUTS:
        return "packed token scheduling requires packed or variable-length layout"

    return None


def _activation_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    recompute = settings.get("activation.recompute")

    if recompute is not None and recompute not in CHECKPOINT_RECOMPUTE_VALUES:
        for key, value in CHECKPOINT_DISABLED_SETTINGS.items():
            if settings.get(key) != value:
                return f"{recompute} requires {key}={value}"

    offload = settings.get("activation.offload")

    if (
        offload in {"saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}
        and fixed_fields.get("activation.saved_tensor_hooks_path") is not True
    ):
        return f"{offload} requires an executable saved-tensor-hooks path"

    return None


def _compile_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    if settings.get("compile.enabled") == "false":
        for key, disabled_value in COMPILE_DISABLED_VALUES.items():
            value = settings.get(key)

            if value is not None and value != disabled_value:
                return f"compile.enabled=false forbids {key}={value}"

    option_values = tuple(settings.get(key) for key in BACKEND_OPTION_KEYS)

    if (
        any(value == "true" for value in option_values)
        and settings.get("compile.mode") is not None
    ):
        return "compile backend options require compile.mode=None"

    if (
        option_values
        and all(value == "false" for value in option_values)
        and settings.get("compile.mode") is None
    ):
        return "disabled compile options forbid compile.mode=None"

    return None


def _reduction_bound_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    if not uses_reduction_degrading_setting(settings):
        return None

    missing = tuple(
        field for field in ADMISSION_BOUND_FIELDS if field not in fixed_fields
    )

    if missing:
        return f"reduction-degrading row missing bound fields: {missing}"

    return None


def _loss_scaling_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    mode = settings.get("numeric.loss_scaling")
    has_scale = "numeric.loss_scale" in settings
    has_degree = "numeric.loss_unscale_degree" in settings

    if mode in {None, "none"}:
        return _disabled_loss_scaling_error(mode, has_scale, has_degree)

    if mode != "static_scale_with_exact_unscale":
        return f"numeric.loss_scaling is unsupported: {mode}"

    return _static_loss_scaling_error(settings, has_scale, has_degree)


def _disabled_loss_scaling_error(
    mode: Any,
    has_scale: bool,
    has_degree: bool,
) -> str | None:
    if not has_scale and not has_degree:
        return None

    if mode is None:
        return "numeric.loss_scaling is required for loss-scale fields"

    return "numeric.loss_scaling=none forbids loss-scale fields"


def _static_loss_scaling_error(
    settings: Mapping[str, Any],
    has_scale: bool,
    has_degree: bool,
) -> str | None:
    if not has_scale:
        return "numeric.loss_scale is required for static loss scaling"

    if not has_degree:
        return "numeric.loss_unscale_degree is required for static loss scaling"

    scale_error = _positive_float_value_error(
        "numeric.loss_scale",
        settings["numeric.loss_scale"],
    )

    if scale_error is not None:
        return scale_error

    degree = settings["numeric.loss_unscale_degree"]

    if isinstance(degree, bool) or degree not in {1, 2}:
        return "numeric.loss_unscale_degree must be 1 or 2"

    return None


def _sampled_fisher_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    sample_source = settings.get("sampled_fisher.sample_source")

    if sample_source is None:
        return None

    if "sampled_fisher.sample_count" not in fixed_fields:
        return "sampled FisherVP requires fixed sample count"

    if (
        sample_source == "fixed_sample_table"
        and "sampled_fisher.sample_table_id" not in fixed_fields
    ):
        return "fixed_sample_table requires sample table identity"

    if (
        sample_source == "fixed_seed_and_count"
        and "sampled_fisher.sample_seed" not in fixed_fields
    ):
        return "fixed_seed_and_count requires sample seed"

    if (
        settings.get("sampled_fisher.exact_fisher_check")
        == "enabled_with_sampling_bound"
        and "sampled_fisher.sampling_bound" not in fixed_fields
    ):
        return "exact Fisher comparison requires sampling-bound formula"

    return None


def _gradient_value_reuse_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    if settings.get("gradient.value_reuse") != "gradient_and_primal_value":
        return None

    if settings.get("gradient.path") == "torch_func_grad_and_value":
        return None

    if fixed_fields.get("gradient.runtime_returns_value_and_grad") is True:
        return None

    return "gradient_and_primal_value requires a value-and-gradient path"


def _jvp_linearize_reuse_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    if settings.get("jvp.linearize_reuse") != "reuse_at_same_primal":
        return None

    if settings.get("jvp.path") == "torch_func_linearize":
        return None

    if fixed_fields.get("jvp.runtime_reuses_linearize_at_same_primal") is True:
        return None

    return "reuse_at_same_primal requires torch_func_linearize"


def _vjp_closure_reuse_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    if settings.get("vjp.closure_reuse") != "reuse_vjp_closure_at_same_primal":
        return None

    if settings.get("vjp.path") == "torch_func_vjp":
        return None

    if fixed_fields.get("vjp.runtime_reuses_closure_at_same_primal") is True:
        return None

    return "reuse_vjp_closure_at_same_primal requires torch_func_vjp"


def _ggn_vjp_path_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    jvp_path = settings.get("ggn.jvp_path")
    vjp_path = settings.get("ggn.vjp_path")

    if jvp_path is None:
        return None

    if vjp_path is None:
        return "ggn.vjp_path is required for JVP-Hessian-VJP rows"

    return None


def _fisher_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    accumulation = settings.get("fisher.accumulation")
    score_path = settings.get("fisher.score_grad_path")

    if "fisher.expectation_path" in settings and (
        settings["fisher.expectation_path"] != "explicit_full_expectation_score_rows"
    ):
        return "fisher.expectation_path is unsupported"

    if accumulation is None:
        return None

    if "fisher.expectation_path" not in settings:
        return "fisher.expectation_path is required for FisherVP rows"

    if accumulation == "streaming_dot_accumulate" and score_path is None:
        return "fisher.score_grad_path is required for streaming rows"

    if (
        accumulation
        in {
            "materialize_score_gradients",
            "blockwise_score_matrix",
        }
        and score_path is not None
    ):
        return f"fisher.score_grad_path is not used with {accumulation}"

    return None


def _iteration_budget_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    solve_path = settings.get("inverse_metric.solve_path")

    if solve_path in ITERATIVE_SOLVE_PATHS:
        if "inverse_metric.iteration_budget" not in settings:
            return "conjugate_gradient requires inverse_metric.iteration_budget"

        if "inverse_metric.preconditioner" not in settings:
            return "conjugate_gradient requires inverse_metric.preconditioner"
    elif "inverse_metric.preconditioner" in settings:
        return "inverse_metric.preconditioner applies only to iterative solves"

    if "inverse_metric.iteration_budget" not in settings:
        return None

    if solve_path not in ITERATIVE_SOLVE_PATHS:
        return "inverse_metric.iteration_budget applies only to iterative solves"

    return None


def _segmented_forward_ad_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    if settings.get("attention.partition") != "segmented_forward_ad":
        return None

    for path_key, values in FORWARD_AD_PATHS_BY_KEY.items():
        if settings.get(path_key) in values:
            return None

    return "attention.partition=segmented_forward_ad requires forward AD path"


def _metric_representation_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    representation = fixed_fields.get("metric.representation")

    if representation is None:
        if _uses_metric_representation(settings):
            return "metric row requires metric.representation fixed field"

        return None

    return _metric_representation_value_error(settings, fixed_fields, representation)


def _uses_metric_representation(settings: Mapping[str, Any]) -> bool:
    return any(
        key.startswith(("metric.", "inverse_metric."))
        or key in {"dtype.metric_factor", "memory.factor_residency"}
        for key in settings
    )


def _metric_representation_value_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
    representation: Any,
) -> str | None:
    multiply_error = _metric_multiply_representation_error(settings, representation)

    if multiply_error is not None:
        return multiply_error

    solve_error = _metric_solve_representation_error(
        settings,
        fixed_fields,
        representation,
    )

    if solve_error is not None:
        return solve_error

    return _metric_factor_setting_error(settings, representation)


def _metric_multiply_representation_error(
    settings: Mapping[str, Any],
    representation: Any,
) -> str | None:
    multiply_path = settings.get("metric.multiply_path")
    requirement_error = _representation_requirement_error(
        multiply_path,
        representation,
        (
            (
                "dense_matmul",
                DENSE_REPRESENTATIONS,
                "metric.multiply_path=dense_matmul requires dense matrix",
            ),
            (
                "factorized_multiply",
                FACTOR_REPRESENTATIONS,
                "metric.multiply_path=factorized_multiply requires factors",
            ),
            (
                "blockwise_multiply",
                BLOCK_REPRESENTATIONS,
                "metric.multiply_path=blockwise_multiply requires blocks",
            ),
            (
                "streaming_multiply",
                STREAMING_REPRESENTATIONS,
                "metric.multiply_path=streaming_multiply requires streamable fields",
            ),
        ),
    )

    if requirement_error is not None:
        return requirement_error

    block_schedule_error = _metric_block_schedule_error(
        settings,
        representation,
        "metric.block_schedule",
    )

    if block_schedule_error is not None:
        return block_schedule_error

    if (
        settings.get("metric.accumulation") is not None
        and settings.get("metric.multiply_path") == "dense_matmul"
    ):
        return "metric.accumulation applies only to non-dense metric paths"

    accumulation_error = _metric_accumulation_error(settings)

    if accumulation_error is not None:
        return accumulation_error

    return None


def _metric_accumulation_error(settings: Mapping[str, Any]) -> str | None:
    value = settings.get("metric.accumulation")
    path = settings.get("metric.multiply_path")

    if path is None:
        if value is not None:
            return "metric.accumulation requires metric.multiply_path"

        return None

    if path == "dense_matmul":
        return None

    expected = "streaming" if path == "streaming_multiply" else "materialized_blocks"

    if value is None:
        return "metric.accumulation is required for non-dense metric paths"

    if value != expected:
        return f"metric.accumulation must be {expected} for this path"

    return None


def _representation_requirement_error(
    path: Any,
    representation: Any,
    requirements: Sequence[tuple[Any, tuple[str, ...], str]],
) -> str | None:
    kind = _metric_representation_kind_field(representation)

    for required_path, representations, message in requirements:
        if path == required_path and kind not in representations:
            return message

    return None


def _metric_solve_representation_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
    representation: Any,
) -> str | None:
    solve_path = settings.get("inverse_metric.solve_path")
    requirement_error = _representation_requirement_error(
        solve_path,
        representation,
        (
            (
                "dense_solve",
                DENSE_REPRESENTATIONS,
                "inverse_metric.solve_path=dense_solve requires dense matrix",
            ),
            (
                "cholesky_solve",
                DENSE_REPRESENTATIONS,
                "inverse_metric.solve_path=cholesky_solve requires dense matrix",
            ),
            (
                "eigh_solve",
                DENSE_REPRESENTATIONS,
                "inverse_metric.solve_path=eigh_solve requires dense matrix",
            ),
            (
                "svd_solve",
                DENSE_REPRESENTATIONS,
                "inverse_metric.solve_path=svd_solve requires dense matrix",
            ),
            (
                "factorized_solve",
                FACTOR_REPRESENTATIONS,
                "factorized_solve requires factors",
            ),
            (
                "blockwise_solve",
                BLOCK_REPRESENTATIONS,
                "blockwise_solve requires blocks",
            ),
            (
                "woodbury_low_rank_solve",
                LOW_RANK_REPRESENTATIONS,
                "woodbury_low_rank_solve requires low-rank factors",
            ),
        ),
    )

    if requirement_error is not None:
        return requirement_error

    field_error = _metric_solve_field_error(settings, fixed_fields)

    if field_error is not None:
        return field_error

    preconditioner_error = _metric_preconditioner_representation_error(
        settings,
        representation,
    )

    if preconditioner_error is not None:
        return preconditioner_error

    block_schedule_error = _metric_block_schedule_error(
        settings,
        representation,
        "inverse_metric.block_schedule",
    )

    if block_schedule_error is not None:
        return block_schedule_error

    return None


def _metric_block_schedule_error(
    settings: Mapping[str, Any],
    representation: Any,
    axis_key: str,
) -> str | None:
    value = settings.get(axis_key)

    if value is None:
        return None

    kind = _metric_representation_kind_field(representation)

    if kind not in BLOCK_OR_KFAC_REPRESENTATIONS:
        return f"{axis_key} requires blocks or KFAC factors"

    if not isinstance(representation, Mapping):
        return f"{axis_key} requires representation.block_schedule"

    schedule = representation.get("block_schedule")

    if not isinstance(schedule, str):
        return f"{axis_key} requires representation.block_schedule"

    if value != schedule:
        return f"{axis_key} must match representation.block_schedule"

    return None


def _metric_representation_kind_field(representation: Any) -> Any:
    if isinstance(representation, Mapping):
        return representation.get("kind")

    return representation


def _metric_solve_field_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    solve_path = settings.get("inverse_metric.solve_path")

    if solve_path == "conjugate_gradient" and "metric.multiply_path" not in settings:
        return "conjugate_gradient requires metric.multiply_path"

    if solve_path == "cholesky_solve" and fixed_fields.get("metric.psd") is not True:
        return "cholesky_solve requires a PSD metric"

    if solve_path == "eigh_solve" and fixed_fields.get("metric.symmetric") is not True:
        return "eigh_solve requires a symmetric metric"

    return None


def _metric_preconditioner_representation_error(
    settings: Mapping[str, Any],
    representation: Any,
) -> str | None:
    preconditioner = settings.get("inverse_metric.preconditioner")
    kind = _metric_representation_kind_field(representation)

    if preconditioner == "block_diagonal" and kind not in BLOCK_OR_KFAC_REPRESENTATIONS:
        return "block_diagonal preconditioner requires blocks or KFAC factors"

    if preconditioner == "factorized_metric" and kind not in FACTOR_REPRESENTATIONS:
        return "factorized_metric preconditioner requires factors"

    return None


def _metric_factor_setting_error(
    settings: Mapping[str, Any],
    representation: Any,
) -> str | None:
    if "dtype.metric_factor" in settings and not _uses_metric_factors(
        settings,
        representation,
    ):
        return "dtype.metric_factor requires a factorized metric path"

    if "memory.factor_residency" in settings and not _uses_metric_factors(
        settings,
        representation,
    ):
        return "memory.factor_residency requires a factorized metric path"

    return None


def _uses_metric_factors(
    settings: Mapping[str, Any],
    representation: Any,
) -> bool:
    if _metric_representation_kind_field(representation) in FACTOR_REPRESENTATIONS:
        return True

    if settings.get("metric.multiply_path") in {
        "factorized_multiply",
        "streaming_multiply",
    }:
        return True

    return settings.get("inverse_metric.solve_path") in {
        "factorized_solve",
        "woodbury_low_rank_solve",
    }


def _tp_loss_parallel_rule_error(
    settings: Mapping[str, Any],
    fixed_fields: Mapping[str, Any],
) -> str | None:
    if settings.get("tp.loss_parallel") != "true":
        return None

    if fixed_fields.get("tp.exact_cross_shard_normalization") is not True:
        return "tp.loss_parallel=true requires exact cross-shard normalization"

    if fixed_fields.get("tp.multi_rank_agreement_check") is not True:
        return "tp.loss_parallel=true requires multi-rank agreement check"

    return None


ATTENTION_FRONTEND_VALUES = (
    "transformers_eager",
    "transformers_sdpa",
    "transformers_flash_attention_2",
    "transformers_flash_attention_3",
    "transformers_flash_attention_4",
    "transformers_flex_attention",
    "paged|eager",
    "paged|sdpa",
    "paged|flash_attention_2",
    "paged|flash_attention_3",
    "paged|flash_attention_4",
    "registered_transformers_attention",
    "pytorch_sdpa_direct",
    "patched_eager",
    "packed_exact",
    "blockwise_exact",
)
SDPA_KERNEL_VALUES = (
    "math",
    "flash_attention",
    "efficient_attention",
    "cudnn_attention",
    "overrideable",
    "priority_list",
)
FORWARD_AD_TRANSFORM_PATHS = ("torch_func_jvp", "jvp_grad")
TORCH_FUNC_AXIS_FIELDS = tuple(
    field for field in TORCH_FUNC_FIELDS if field not in FORWARD_AD_FIELDS
)
VMAP_TRANSFORM_PATHS = ("per_example_gradient_vmap",)
VMAP_PATH_SETTINGS = (
    ("fisher.score_grad_path", "vmap_grad"),
    ("sampled_fisher.score_grad_path", "vmap_grad"),
    ("empirical_fisher.grad_path", "vmap_grad"),
)
PER_EXAMPLE_LOOP_PATHS = (
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "backward_materialized_grad",
)
HVP_VECTOR_LOOP_PATHS = (
    "reverse_over_reverse",
    "jvp_grad",
    "autograd_functional_hvp",
    "autograd_functional_vhp",
    "forward_ad_dual",
    "linearize_grad",
)
HVP_VECTOR_VMAP_PATHS = ("linearize_grad",)
JVP_VECTOR_LOOP_PATHS = ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize")
JVP_VECTOR_VMAP_PATHS = ("torch_func_jvp", "torch_func_linearize")
VJP_VECTOR_LOOP_PATHS = (
    "torch_func_vjp",
    "autograd_grad_outputs",
    "backward_materialized_grad",
)
VJP_VECTOR_VMAP_PATHS = ("torch_func_vjp",)
GGN_VECTOR_LOOP_PATHS = ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize")
GGN_VECTOR_VMAP_PATHS = ("torch_func_jvp", "torch_func_linearize")
FISHER_VECTOR_ACCUMULATIONS = (
    "streaming_dot_accumulate",
    "materialize_score_gradients",
    "blockwise_score_matrix",
)
SAMPLED_FISHER_VECTOR_ACCUMULATIONS = FISHER_VECTOR_ACCUMULATIONS
EMPIRICAL_FISHER_VECTOR_GRAD_PATHS = (
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "vmap_grad",
    "backward_materialized_grad",
)
EMPIRICAL_FISHER_VECTOR_ACCUMULATIONS = (
    "materialize_per_example_gradients",
    "blockwise_gradient_matrix",
)
MATMUL_PRECISION_VALUES = ("highest", "high", "medium")
SPEC_DTYPE_VALUES = ("fp32", "bf16", "fp16")


@dataclasses.dataclass(frozen=True, slots=True)
class AxisDescriptor:
    """One tunable axis registered by core or an adapter."""

    name: str
    settings_keys: tuple[str, ...]
    allowed_values: tuple[Any, ...]
    optional_settings_keys: tuple[str, ...] = ()
    adapter_id: str = "core"
    adapter_version: str = "0.0.1"
    admission_rule: AdmissionRule | None = None
    identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def admit(self, candidate: Candidate) -> tuple[bool, str | None]:
        """Run the admission rule for a candidate.

        Returns:
            Admission status and optional failure reason.
        """
        value_error = _axis_value_error(self, candidate)

        if value_error is not None:
            return False, value_error

        if self.admission_rule is None:
            return True, None

        return self.admission_rule(candidate)

    def signature(self) -> dict[str, Any]:
        """Return stable axis identity."""
        return {
            "name": self.name,
            "settings_keys": self.settings_keys,
            "optional_settings_keys": self.optional_settings_keys,
            "allowed_values": self.allowed_values,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "has_admission_rule": self.admission_rule is not None,
            "identity": dict(self.identity),
        }


@dataclasses.dataclass(slots=True)
class AxisRegistry:
    """Registry that prevents overlapping setting ownership."""

    axes: dict[str, AxisDescriptor] = dataclasses.field(default_factory=dict)
    owners: dict[str, str] = dataclasses.field(default_factory=dict)
    optional_owners: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )

    def register(self, axis: AxisDescriptor) -> None:
        """Register an axis descriptor.

        Raises:
            AdmissionError: If the axis has no validator, or if the axis or any
                owned setting key is duplicated.
        """
        if not axis.allowed_values and axis.admission_rule is None:
            message = (
                f"axis must declare allowed values or an admission rule: {axis.name}"
            )
            raise AdmissionError(message)

        if axis.name in self.axes:
            message = f"axis is already registered: {axis.name}"
            raise AdmissionError(message)

        all_setting_keys = (*axis.settings_keys, *axis.optional_settings_keys)

        if len(set(all_setting_keys)) != len(all_setting_keys):
            message = f"axis setting keys are duplicated: {axis.name}"
            raise AdmissionError(message)

        for key in axis.settings_keys:
            owner = self.owners.get(key)

            if owner is not None:
                message = f"setting key has multiple axis owners: {key}"
                raise AdmissionError(message)

        self.axes[axis.name] = axis

        for key in axis.settings_keys:
            self.owners[key] = axis.name

        for key in axis.optional_settings_keys:
            owners = self.optional_owners.get(key, ())

            if axis.name not in owners:
                self.optional_owners[key] = (*owners, axis.name)

    def admit(self, candidate: Candidate) -> Candidate:
        """Return candidate with admission status set.

        Returns:
            Candidate with updated admission status.

        Raises:
            AdmissionError: If a changed axis is unknown.
        """
        axis_names = set(candidate.changed_axes)

        for key in candidate.settings:
            owner = self.owners.get(key)

            if owner is not None:
                axis_names.add(owner)
                continue

            optional_owners = self.optional_owners.get(key)

            if optional_owners is None:
                message = f"candidate setting key has no axis owner: {key}"
                raise AdmissionError(message)

            axis_names.update(optional_owners)

        for axis_name in sorted(axis_names):
            axis = self.axes.get(axis_name)

            if axis is None:
                message = f"candidate axis is unknown: {axis_name}"
                raise AdmissionError(message)

            passed, reason = axis.admit(candidate)

            if not passed:
                return dataclasses.replace(
                    candidate,
                    admission_status="failed",
                    admission_error=reason or f"axis rejected: {axis_name}",
                )

        return dataclasses.replace(candidate, admission_status="passed")

    def signature(self) -> dict[str, Any]:
        """Return stable registry identity."""
        return {
            "axes": {
                name: axis.signature() for name, axis in sorted(self.axes.items())
            },
            "owners": dict(sorted(self.owners.items())),
            "optional_owners": {
                key: tuple(value) for key, value in sorted(self.optional_owners.items())
            },
        }


def topological_families(families: Sequence[Family]) -> tuple[Family, ...]:
    """Return families in dependency order.

    Raises:
        RuntimeError: If names are duplicated, missing, or cyclic.
    """
    by_name = {family.name: family for family in families}

    if len(by_name) != len(families):
        message = "family names must be unique"
        raise RuntimeError(message)

    ordered = []
    visiting = set()
    visited = set()

    def visit(name: str) -> None:
        if name in visited:
            return

        if name in visiting:
            message = f"family DAG has a cycle at {name}"
            raise RuntimeError(message)

        family = by_name.get(name)

        if family is None:
            message = f"family dependency is missing: {name}"
            raise RuntimeError(message)

        visiting.add(name)

        for dependency in family.dependencies:
            visit(dependency)

        visiting.remove(name)
        visited.add(name)
        ordered.append(family)

    for family in families:
        visit(family.name)

    return tuple(ordered)


def _axis_value_error(axis: AxisDescriptor, candidate: Candidate) -> str | None:
    missing = tuple(key for key in axis.settings_keys if key not in candidate.settings)

    if missing:
        return f"candidate missing axis setting: {axis.name}"

    if len(axis.settings_keys) == 1:
        value = candidate.settings[axis.settings_keys[0]]
    else:
        value = {key: candidate.settings[key] for key in axis.settings_keys}

    if not axis.allowed_values or any(
        value == allowed for allowed in axis.allowed_values
    ):
        return None

    return f"candidate axis value is not allowed: {axis.name}"


def _positive_int_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            error = _positive_int_error(candidate.settings, key)

            if error is not None:
                return False, error

        return True, None

    return admit


def _positive_int_error(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings[key]

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return f"candidate axis must be a positive integer: {key}"

    return None


def _bool_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            if not isinstance(candidate.settings[key], bool):
                return False, f"candidate axis must be boolean: {key}"

        return True, None

    return admit


def _gradient_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["gradient.path"]

        if value in {"torch_func_grad", "torch_func_grad_and_value"}:
            return _admit_torch_func_path("torch_func_vjp", candidate.settings)

        return True, None

    return admit


def _gradient_value_reuse_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["gradient.value_reuse"]

        if (
            value == "gradient_and_primal_value"
            and candidate.settings.get("gradient.path") != "torch_func_grad_and_value"
        ):
            return (
                False,
                "gradient_and_primal_value requires torch_func_grad_and_value",
            )

        return True, None

    return admit


def _jvp_linearize_reuse_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["jvp.linearize_reuse"]

        if (
            value == "reuse_at_same_primal"
            and candidate.settings.get("jvp.path") != "torch_func_linearize"
        ):
            return False, "reuse_at_same_primal requires torch_func_linearize"

        return True, None

    return admit


def _vjp_closure_reuse_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["vjp.closure_reuse"]

        if (
            value == "reuse_vjp_closure_at_same_primal"
            and candidate.settings.get("vjp.path") != "torch_func_vjp"
        ):
            return False, "reuse_vjp_closure_at_same_primal requires torch_func_vjp"

        return True, None

    return admit


def _jvp_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["jvp.path"]

        if value == "forward_ad_dual":
            return _admit_forward_ad_path(candidate.settings)

        return _admit_torch_func_path("torch_func_jvp", candidate.settings)

    return admit


def _vjp_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["vjp.path"]

        if value == "torch_func_vjp":
            return _admit_torch_func_path(value, candidate.settings)

        return True, None

    return admit


def _hvp_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["hvp.path"]

        if value == "forward_ad_dual":
            return _admit_forward_ad_path(candidate.settings)

        if value == "jvp_grad":
            return _admit_torch_func_path(value, candidate.settings)

        if value == "linearize_grad":
            return _admit_torch_func_path("jvp_grad", candidate.settings)

        return True, None

    return admit


def _ggn_jvp_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["ggn.jvp_path"]

        if value == "torch_func_jvp":
            return _admit_torch_func_path(value, candidate.settings)

        if value == "torch_func_linearize":
            return _admit_torch_func_path("torch_func_jvp", candidate.settings)

        if value == "forward_ad_dual":
            return _admit_forward_ad_path(candidate.settings)

        return True, None

    return admit


def _ggn_vjp_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["ggn.vjp_path"]

        if value == "torch_func_vjp":
            return _admit_torch_func_path(value, candidate.settings)

        return True, None

    return admit


def _fisher_score_grad_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["fisher.score_grad_path"]

        if value == "torch_func_grad":
            return _admit_torch_func_path("torch_func_vjp", candidate.settings)

        if value == "vmap_grad":
            return _admit_torch_func_path(
                "per_example_gradient_vmap", candidate.settings
            )

        return True, None

    return admit


def _empirical_fisher_grad_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["empirical_fisher.grad_path"]

        if value == "vmap_grad":
            return _admit_torch_func_path(
                "per_example_gradient_vmap", candidate.settings
            )

        if value == "torch_func_grad":
            return _admit_torch_func_path("torch_func_vjp", candidate.settings)

        return True, None

    return admit


def _composition_execution_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["composition.execution"]

        if (
            value == "compile_whole_composition"
            and candidate.settings.get("compile.enabled") != "true"
        ):
            return (
                False,
                "compile_whole_composition requires compile.enabled=true",
            )

        return True, None

    return admit


def _loss_scaling_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _loss_scaling_rule_error(candidate.settings, {})

        if error is not None:
            return False, error

        return True, None

    return admit


def _sampled_fisher_score_grad_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["sampled_fisher.score_grad_path"]

        if value == "torch_func_grad":
            return _admit_torch_func_path("torch_func_vjp", candidate.settings)

        if value == "vmap_grad":
            return _admit_torch_func_path(
                "per_example_gradient_vmap", candidate.settings
            )

        return True, None

    return admit


def _admit_forward_ad_path(settings: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        admit_forward_ad(settings)
    except AdmissionError as error:
        return False, str(error)

    return True, None


def _admit_torch_func_path(
    value: str,
    settings: Mapping[str, Any],
) -> tuple[bool, str | None]:
    missing = tuple(field for field in TORCH_FUNC_FIELDS if field not in settings)

    if missing:
        return False, f"torch.func admission fields missing: {missing}"

    if value in VMAP_TRANSFORM_PATHS:
        vmap_error = _vmap_path_error(settings)

        if vmap_error is not None:
            return False, vmap_error

    try:
        admit_torch_func(settings)

        if value in FORWARD_AD_TRANSFORM_PATHS:
            admit_forward_ad(settings)
    except AdmissionError as error:
        return False, str(error)

    return True, None


def _vmap_path_error(settings: Mapping[str, Any]) -> str | None:
    if settings["requires_forward_ad"] is True:
        return "per_example_gradient_vmap does not use forward AD"

    if settings.get("schedule.per_example") != "vmap":
        return "vmap_grad rows require schedule.per_example=vmap"

    return None


def _per_example_schedule_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _per_example_schedule_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _per_example_schedule_error(settings: Mapping[str, Any]) -> str | None:
    schedule = settings["schedule.per_example"]
    score_path = _per_example_score_path(settings)

    if schedule == "vmap":
        error = (
            None
            if score_path == "vmap_grad"
            else "schedule.per_example=vmap requires vmap_grad"
        )
    elif schedule == "loop":
        error = (
            None
            if score_path in PER_EXAMPLE_LOOP_PATHS
            else "schedule.per_example=loop requires a loop score-gradient path"
        )
    elif schedule == "manual_batch":
        if score_path in PER_EXAMPLE_LOOP_PATHS:
            error = _manual_per_example_batch_size_error(settings)
        else:
            error = (
                "schedule.per_example=manual_batch requires a loop score-gradient path"
            )
    else:
        error = f"schedule.per_example is unsupported: {schedule}"

    return error


def _per_example_score_path(settings: Mapping[str, Any]) -> str | None:
    if "fisher.score_grad_path" in settings:
        return settings["fisher.score_grad_path"]

    if "sampled_fisher.score_grad_path" in settings:
        return settings["sampled_fisher.score_grad_path"]

    if "empirical_fisher.grad_path" in settings:
        return settings["empirical_fisher.grad_path"]

    return None


def _manual_per_example_batch_size_error(settings: Mapping[str, Any]) -> str | None:
    if "empirical_fisher.grad_path" in settings:
        key = "batch.empirical_example_batch_size"
    elif (
        "fisher.score_grad_path" in settings
        or "sampled_fisher.score_grad_path" in settings
    ):
        key = "batch.fisher_sample_batch_size"
    else:
        return "schedule.per_example=manual_batch requires a Fisher-family path"

    if key not in settings:
        return f"{key} is required for schedule.per_example=manual_batch"

    value = settings[key]

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return f"{key} must be a positive integer"

    return None


def _uses_vmap_path(settings: Mapping[str, Any]) -> bool:
    return any(settings.get(key) == value for key, value in VMAP_PATH_SETTINGS)


def _uses_hvp_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("hvp.path") in HVP_VECTOR_LOOP_PATHS


def _uses_hvp_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("hvp.path") in HVP_VECTOR_VMAP_PATHS


def _uses_jvp_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("jvp.path") in JVP_VECTOR_LOOP_PATHS


def _uses_jvp_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("jvp.path") in JVP_VECTOR_VMAP_PATHS


def _uses_vjp_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("vjp.path") in VJP_VECTOR_LOOP_PATHS


def _uses_vjp_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("vjp.path") in VJP_VECTOR_VMAP_PATHS


def _uses_ggn_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("ggn.jvp_path") in GGN_VECTOR_LOOP_PATHS and settings.get(
        "ggn.vjp_path"
    ) in {
        "torch_func_vjp",
        "autograd_grad_outputs",
    }


def _uses_ggn_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("ggn.jvp_path") in GGN_VECTOR_VMAP_PATHS
        and settings.get("ggn.vjp_path") == "torch_func_vjp"
    )


def _uses_fisher_vector_path(settings: Mapping[str, Any]) -> bool:
    return settings.get("fisher.accumulation") in FISHER_VECTOR_ACCUMULATIONS


def _uses_sampled_fisher_vector_path(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("sampled_fisher.accumulation")
        in SAMPLED_FISHER_VECTOR_ACCUMULATIONS
    )


def _uses_empirical_fisher_vector_path(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("empirical_fisher.grad_path") in EMPIRICAL_FISHER_VECTOR_GRAD_PATHS
        or settings.get("empirical_fisher.accumulation")
        in EMPIRICAL_FISHER_VECTOR_ACCUMULATIONS
    )


def _uses_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return (
        _uses_jvp_vector_loop_path(settings)
        or _uses_vjp_vector_loop_path(settings)
        or _uses_ggn_vector_loop_path(settings)
        or _uses_hvp_vector_loop_path(settings)
        or _uses_fisher_vector_path(settings)
        or _uses_sampled_fisher_vector_path(settings)
        or _uses_empirical_fisher_vector_path(settings)
        or settings.get("composition.execution")
        in {
            "materialize_each_child",
            "stream_child_outputs",
            "fuse_adjacent_children",
            "compile_whole_composition",
        }
    )


def _uses_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return (
        _uses_jvp_vector_vmap_path(settings)
        or _uses_vjp_vector_vmap_path(settings)
        or _uses_ggn_vector_vmap_path(settings)
        or _uses_hvp_vector_vmap_path(settings)
        or _uses_fisher_vector_path(settings)
        or _uses_sampled_fisher_vector_path(settings)
        or _uses_empirical_fisher_vector_path(settings)
        or settings.get("composition.execution")
        in {
            "materialize_each_child",
            "stream_child_outputs",
            "fuse_adjacent_children",
            "compile_whole_composition",
        }
    )


def _vectorization_mode_error(settings: Mapping[str, Any]) -> str | None:
    mode_key = "vectorization.mode"
    uses_vector_vmap = _uses_vector_vmap_path(settings)
    uses_vector_loop = _uses_vector_loop_path(settings)

    if mode_key not in settings:
        return None

    mode = settings[mode_key]
    error = None

    if mode == "manual_batch" and uses_vector_loop:
        error = _manual_batch_size_error(settings) or _vmap_batch_in_dims_error(
            settings
        )
    elif mode == "manual_batch":
        error = "vectorization.mode=manual_batch requires vector-loop lowering"
    elif mode == "vmap" and uses_vector_vmap:
        error = (
            _vmap_chunk_size_error(settings)
            or _vmap_batch_in_dims_error(settings)
            or _vmap_randomness_error(settings)
        )
    elif mode == "vmap":
        error = "vectorization.mode=vmap requires vector-axis vmap lowering"
    elif mode == "single_loop" and uses_vector_loop:
        error = _vmap_batch_in_dims_error(settings)
    elif mode == "single_loop":
        error = "vectorization.mode=single_loop requires vector-loop lowering"

    return error


def _vectorization_mode_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _vectorization_mode_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _vmap_chunk_size_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.vmap_chunk_size"

    if key not in settings:
        return "vectorization.mode=vmap requires vectorization.vmap_chunk_size"

    chunk_size = settings[key]

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        return "vectorization.vmap_chunk_size must be a positive integer"

    return None


def _manual_batch_size_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.batch_size"

    if key not in settings:
        return "vectorization.mode=manual_batch requires vectorization.batch_size"

    batch_size = settings[key]

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        return "vectorization.batch_size must be a positive integer"

    return None


def _gradient_accumulation_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _gradient_accumulation_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _gradient_accumulation_error(settings: Mapping[str, Any]) -> str | None:
    schedule_key = "schedule.gradient_accumulation"
    batch_size_key = "batch.data_microbatch_size"
    schedule = settings.get(schedule_key)
    has_batch_size = batch_size_key in settings
    error = None

    if schedule in {None, "single_step"} and has_batch_size:
        error = f"{batch_size_key} requires {schedule_key}=microbatch_accumulate"
    elif schedule in {None, "single_step"}:
        error = None
    elif schedule != "microbatch_accumulate":
        error = f"{schedule_key} is unsupported"
    elif not any(key in settings for key in MICROBATCH_OPERATOR_PATH_KEYS):
        error = f"{schedule_key}=microbatch_accumulate requires operator path"
    elif (
        settings.get("compile.enabled") == "true"
        and settings.get("compile.boundary") == "loss_closure"
    ):
        error = "loss_closure compile boundary is incompatible with microbatching"
    elif not has_batch_size:
        error = f"{schedule_key}=microbatch_accumulate requires {batch_size_key}"
    else:
        error = _positive_int_error(settings, batch_size_key)

    return error


def _vmap_randomness_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.randomness"

    if key not in settings:
        return "vectorization.mode=vmap requires vectorization.randomness"

    if settings[key] not in {"error", "same", "different"}:
        return "vectorization.randomness is unsupported"

    return None


def _vmap_batch_in_dims_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.in_dims"

    if key not in settings:
        return "vectorized rows require vectorization.in_dims"

    return _vmap_batch_in_dims_value_error(settings[key])


def _vmap_batch_in_dims_value_error(in_dims: Any) -> str | None:
    if not isinstance(in_dims, Mapping) or not in_dims:
        return "vectorization.in_dims must be a nonempty mapping"

    for key, value in in_dims.items():
        if not isinstance(key, str) or not key:
            return "vectorization.in_dims keys must be nonempty strings"

        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return "vectorization.in_dims values must be integers or None"

    return None


def _vmap_chunk_size_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if candidate.settings.get("vectorization.mode") != "vmap":
            return (
                False,
                "vectorization.vmap_chunk_size requires vectorization.mode=vmap",
            )

        error = _vmap_chunk_size_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _vmap_batch_in_dims_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if "vectorization.mode" not in candidate.settings:
            return False, "vectorization.in_dims requires vectorization.mode"

        error = _vmap_batch_in_dims_value_error(
            candidate.settings["vectorization.in_dims"]
        )

        return (True, None) if error is None else (False, error)

    return admit


def _torch_func_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        try:
            admit_torch_func(candidate.settings)
        except AdmissionError as error:
            return False, str(error)

        return True, None

    return admit


def standard_axis_descriptors() -> tuple[AxisDescriptor, ...]:
    """Return standard core axis descriptors."""
    return (
        AxisDescriptor(
            "dtype.parameter_storage",
            ("dtype.parameter_storage",),
            SPEC_DTYPE_VALUES,
        ),
        AxisDescriptor(
            "dtype.model_compute",
            ("dtype.model_compute",),
            SPEC_DTYPE_VALUES,
        ),
        AxisDescriptor("dtype.vector", ("dtype.vector",), SPEC_DTYPE_VALUES),
        AxisDescriptor(
            "dtype.intermediate",
            ("dtype.intermediate",),
            SPEC_DTYPE_VALUES,
        ),
        AxisDescriptor(
            "dtype.metric_factor",
            ("dtype.metric_factor",),
            SPEC_DTYPE_VALUES,
        ),
        AxisDescriptor("dtype.output", ("dtype.output",), SPEC_DTYPE_VALUES),
        AxisDescriptor(
            "layout.params",
            ("layout.params",),
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        AxisDescriptor(
            "layout.vector",
            ("layout.vector",),
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        AxisDescriptor(
            "layout.output",
            ("layout.output",),
            (
                "parameter_tree",
                "flat_contiguous",
                "per_layer_flat",
                "per_block_flat",
                "per_shard",
                "dtensor",
            ),
        ),
        AxisDescriptor(
            "layout.contiguity",
            ("layout.contiguity",),
            ("contiguous", "preserve_existing_strides"),
        ),
        AxisDescriptor(
            "layout.flatten_order",
            ("layout.flatten_order",),
            ("canonical_parameter_order",),
        ),
        AxisDescriptor(
            "layout.vector_ops",
            ("layout.vector_ops",),
            ("python_loop", "foreach"),
        ),
        AxisDescriptor(
            "layout.aliasing",
            ("layout.aliasing",),
            ("preserve_tied_weight_aliases",),
        ),
        AxisDescriptor(
            "layout.parametrizations",
            ("layout.parametrizations",),
            ("preserve_active_parametrizations",),
        ),
        AxisDescriptor(
            "call.path", ("call.path",), ("functional_call", "stateful_module")
        ),
        AxisDescriptor(
            "call.params", ("call.params",), ("explicit_params", "module_params")
        ),
        AxisDescriptor(
            "call.buffers",
            ("call.buffers",),
            ("explicit_buffers", "module_buffers"),
        ),
        AxisDescriptor(
            "call.tied_weights",
            ("call.tied_weights",),
            ("preserve_alias_groups",),
        ),
        AxisDescriptor(
            "call.parametrizations",
            ("call.parametrizations",),
            ("preserve_parametrizations",),
        ),
        AxisDescriptor(
            "call.buffer_mutation",
            ("call.buffer_mutation",),
            ("forbidden", "declared_and_restored"),
        ),
        AxisDescriptor("call.grad_mode", ("call.grad_mode",), ("grad_enabled",)),
        AxisDescriptor(
            "call.return_type",
            ("call.return_type",),
            ("raw_tensor_tree", "model_output_object_with_declared_fields"),
        ),
        AxisDescriptor(
            "hvp.path",
            ("hvp.path",),
            (
                "reverse_over_reverse",
                "autograd_functional_hvp",
                "autograd_functional_vhp",
                "jvp_grad",
                "forward_ad_dual",
                "linearize_grad",
            ),
            admission_rule=_hvp_path_axis(),
        ),
        AxisDescriptor(
            "hvp.graph_schedule",
            ("hvp.graph_schedule",),
            ("retain_graph_across_vectors", "rebuild_graph_per_vector"),
        ),
        AxisDescriptor(
            "hvp.primal_reuse",
            ("hvp.primal_reuse",),
            ("reuse_primal", "recompute_primal"),
        ),
        AxisDescriptor(
            "hvp.gradient_reuse",
            ("hvp.gradient_reuse",),
            ("reuse_gradient_closure", "recompute_gradient"),
        ),
        AxisDescriptor(
            "gradient.path",
            ("gradient.path",),
            (
                "torch_autograd_grad",
                "torch_func_grad",
                "torch_func_grad_and_value",
                "backward_materialized_grad",
            ),
            admission_rule=_gradient_path_axis(),
        ),
        AxisDescriptor(
            "gradient.value_reuse",
            ("gradient.value_reuse",),
            ("gradient_only", "gradient_and_primal_value"),
            optional_settings_keys=("gradient.path",),
            admission_rule=_gradient_value_reuse_axis(),
        ),
        AxisDescriptor(
            "jvp.path",
            ("jvp.path",),
            ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize"),
            admission_rule=_jvp_path_axis(),
        ),
        AxisDescriptor(
            "jvp.linearize_reuse",
            ("jvp.linearize_reuse",),
            ("none", "reuse_at_same_primal"),
            optional_settings_keys=("jvp.path",),
            admission_rule=_jvp_linearize_reuse_axis(),
        ),
        AxisDescriptor(
            "vjp.path",
            ("vjp.path",),
            ("torch_func_vjp", "autograd_grad_outputs", "backward_materialized_grad"),
            admission_rule=_vjp_path_axis(),
        ),
        AxisDescriptor(
            "vjp.closure_reuse",
            ("vjp.closure_reuse",),
            ("none", "reuse_vjp_closure_at_same_primal"),
            optional_settings_keys=("vjp.path",),
            admission_rule=_vjp_closure_reuse_axis(),
        ),
        AxisDescriptor(
            "ggn.jvp_path",
            ("ggn.jvp_path",),
            ("torch_func_jvp", "forward_ad_dual", "torch_func_linearize"),
            admission_rule=_ggn_jvp_path_axis(),
        ),
        AxisDescriptor(
            "ggn.loss_hessian_path",
            ("ggn.loss_hessian_path",),
            ("closed_form_softmax_ce_kl", "autodiff_loss_hvp"),
        ),
        AxisDescriptor(
            "ggn.loss_hessian_kernel",
            ("ggn.loss_hessian_kernel",),
            ("dense_global", "streaming_global", "two_pass_chunked_global"),
        ),
        AxisDescriptor(
            "ggn.vjp_path",
            ("ggn.vjp_path",),
            ("torch_func_vjp", "autograd_grad_outputs"),
            optional_settings_keys=("ggn.jvp_path",),
            admission_rule=_ggn_vjp_path_axis(),
        ),
        AxisDescriptor(
            "ggn.jvp_reuse",
            ("ggn.jvp_reuse",),
            ("reuse_jvp", "recompute_jvp"),
        ),
        AxisDescriptor(
            "ggn.cotangent_reuse",
            ("ggn.cotangent_reuse",),
            ("reuse_output_cotangent", "recompute_output_cotangent"),
        ),
        AxisDescriptor(
            "fisher.accumulation",
            ("fisher.accumulation",),
            (
                "streaming_dot_accumulate",
                "materialize_score_gradients",
                "blockwise_score_matrix",
            ),
        ),
        AxisDescriptor(
            "fisher.expectation_path",
            ("fisher.expectation_path",),
            ("explicit_full_expectation_score_rows",),
        ),
        AxisDescriptor(
            "fisher.score_grad_path",
            ("fisher.score_grad_path",),
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
            admission_rule=_fisher_score_grad_path_axis(),
        ),
        AxisDescriptor(
            "sampled_fisher.accumulation",
            ("sampled_fisher.accumulation",),
            (
                "streaming_dot_accumulate",
                "materialize_score_gradients",
                "blockwise_score_matrix",
            ),
        ),
        AxisDescriptor(
            "sampled_fisher.sample_source",
            ("sampled_fisher.sample_source",),
            (
                "fixed_sample_table",
                "fixed_seed_and_count",
            ),
        ),
        AxisDescriptor(
            "sampled_fisher.score_grad_path",
            ("sampled_fisher.score_grad_path",),
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
            admission_rule=_sampled_fisher_score_grad_path_axis(),
        ),
        AxisDescriptor(
            "sampled_fisher.exact_fisher_check",
            ("sampled_fisher.exact_fisher_check",),
            ("disabled", "enabled_with_sampling_bound"),
        ),
        AxisDescriptor(
            "empirical_fisher.grad_path",
            ("empirical_fisher.grad_path",),
            (
                "torch_autograd_grad_loop",
                "torch_func_grad",
                "vmap_grad",
                "backward_materialized_grad",
            ),
            admission_rule=_empirical_fisher_grad_path_axis(),
        ),
        AxisDescriptor(
            "empirical_fisher.accumulation",
            ("empirical_fisher.accumulation",),
            (
                "streaming_dot_accumulate",
                "materialize_per_example_gradients",
                "blockwise_gradient_matrix",
            ),
        ),
        AxisDescriptor(
            "forward_ad_flags",
            FORWARD_AD_FIELDS,
            (),
            admission_rule=_bool_axis(*FORWARD_AD_FIELDS),
        ),
        AxisDescriptor(
            "torch_func_admission",
            TORCH_FUNC_AXIS_FIELDS,
            (),
            admission_rule=_torch_func_axis(),
        ),
        AxisDescriptor(
            "vectorization.mode",
            ("vectorization.mode",),
            ("single_loop", "manual_batch", "vmap"),
            admission_rule=_vectorization_mode_axis(),
        ),
        AxisDescriptor(
            "vectorization.batch_size",
            ("vectorization.batch_size",),
            (),
            admission_rule=_positive_int_axis("vectorization.batch_size"),
        ),
        AxisDescriptor(
            "vectorization.vmap_chunk_size",
            ("vectorization.vmap_chunk_size",),
            (),
            admission_rule=_vmap_chunk_size_axis(),
        ),
        AxisDescriptor(
            "vectorization.in_dims",
            ("vectorization.in_dims",),
            (),
            admission_rule=_vmap_batch_in_dims_axis(),
        ),
        AxisDescriptor(
            "batch.data_microbatch_size",
            ("batch.data_microbatch_size",),
            (),
            admission_rule=_gradient_accumulation_axis(),
        ),
        AxisDescriptor(
            "batch.fisher_sample_batch_size",
            ("batch.fisher_sample_batch_size",),
            (),
            admission_rule=_positive_int_axis("batch.fisher_sample_batch_size"),
        ),
        AxisDescriptor(
            "batch.empirical_example_batch_size",
            ("batch.empirical_example_batch_size",),
            (),
            admission_rule=_positive_int_axis("batch.empirical_example_batch_size"),
        ),
        AxisDescriptor(
            "schedule.per_example",
            ("schedule.per_example",),
            ("loop", "vmap", "manual_batch"),
            admission_rule=_per_example_schedule_axis(),
        ),
        AxisDescriptor(
            "schedule.gradient_accumulation",
            ("schedule.gradient_accumulation",),
            ("single_step", "microbatch_accumulate"),
            admission_rule=_gradient_accumulation_axis(),
        ),
        AxisDescriptor(
            "schedule.per_token", ("schedule.per_token",), ("loop", "packed")
        ),
        AxisDescriptor(
            "input.batch_layout",
            ("input.batch_layout",),
            ("dense_padded", "packed_with_inverse_permutation", "variable_length"),
        ),
        AxisDescriptor(
            "input.length_grouping",
            ("input.length_grouping",),
            ("none", "exact_length_bucket"),
        ),
        AxisDescriptor(
            "input.host_to_device",
            ("input.host_to_device",),
            ("outside_measured_call", "inside_measured_call"),
        ),
        AxisDescriptor(
            "input.residency",
            ("input.residency",),
            ("cpu_staged", "cpu_pinned", "gpu"),
        ),
        AxisDescriptor(
            "teacher_outputs",
            ("teacher_outputs",),
            (
                "precomputed_cpu",
                "precomputed_cpu_pinned",
                "precomputed_gpu",
                "recomputed_with_equality_check",
            ),
        ),
        AxisDescriptor(
            "memory.vector_residency",
            ("memory.vector_residency",),
            ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu"),
        ),
        AxisDescriptor(
            "memory.intermediate_residency",
            ("memory.intermediate_residency",),
            ("gpu", "cpu_pinned", "cpu_staged"),
        ),
        AxisDescriptor(
            "memory.factor_residency",
            ("memory.factor_residency",),
            ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu"),
        ),
        AxisDescriptor(
            "memory.output_buffers",
            ("memory.output_buffers",),
            ("fresh_allocation", "preallocated"),
        ),
        AxisDescriptor(
            "chunk.class_block_size_with_exact_global_normalization",
            ("chunk.class_block_size_with_exact_global_normalization",),
            (),
            admission_rule=_positive_int_axis(
                "chunk.class_block_size_with_exact_global_normalization",
            ),
        ),
        AxisDescriptor("compile.enabled", ("compile.enabled",), ("false", "true")),
        AxisDescriptor(
            "compile.boundary",
            ("compile.boundary",),
            ("whole_operator",),
        ),
        AxisDescriptor("compile.backend", ("compile.backend",), ("inductor",)),
        AxisDescriptor(
            "compile.mode",
            ("compile.mode",),
            (None, "default", "max-autotune"),
        ),
        AxisDescriptor("compile.fullgraph", ("compile.fullgraph",), ("false", "true")),
        AxisDescriptor(
            "compile.dynamic",
            ("compile.dynamic",),
            (None, "false", "true"),
        ),
        AxisDescriptor(
            "compile.compiled_autograd",
            ("compile.compiled_autograd",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "compile.options.epilogue_fusion",
            ("compile.options.epilogue_fusion",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "compile.options.shape_padding",
            ("compile.options.shape_padding",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "compile.cuda_graphs",
            ("compile.cuda_graphs",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "compile.cache_state",
            ("compile.cache_state",),
            ("cold_compile", "warm_cache"),
        ),
        AxisDescriptor(
            "numeric.float32_matmul_precision",
            ("numeric.float32_matmul_precision",),
            MATMUL_PRECISION_VALUES,
        ),
        AxisDescriptor(
            "autocast",
            ("autocast",),
            ("off", "cuda_fp16", "cuda_bf16"),
        ),
        AxisDescriptor(
            "metric.multiply_path",
            ("metric.multiply_path",),
            (
                "dense_matmul",
                "factorized_multiply",
                "blockwise_multiply",
                "streaming_multiply",
            ),
        ),
        AxisDescriptor(
            "metric.accumulation",
            ("metric.accumulation",),
            ("streaming", "materialized_blocks"),
            optional_settings_keys=("metric.multiply_path",),
        ),
        AxisDescriptor(
            "inverse_metric.solve_path",
            ("inverse_metric.solve_path",),
            (
                "dense_solve",
                "cholesky_solve",
                "eigh_solve",
                "svd_solve",
                "conjugate_gradient",
                "factorized_solve",
                "blockwise_solve",
                "woodbury_low_rank_solve",
            ),
        ),
        AxisDescriptor(
            "inverse_metric.preconditioner",
            ("inverse_metric.preconditioner",),
            ("none", "diagonal", "block_diagonal", "factorized_metric"),
        ),
        AxisDescriptor(
            "inverse_metric.iteration_budget",
            ("inverse_metric.iteration_budget",),
            (),
            admission_rule=_positive_int_axis("inverse_metric.iteration_budget"),
        ),
        AxisDescriptor(
            "inverse_metric.factor_reuse",
            ("inverse_metric.factor_reuse",),
            ("refactor_each_rhs", "reuse_factor_across_rhs"),
        ),
        AxisDescriptor(
            "composition.execution",
            ("composition.execution",),
            (
                "materialize_each_child",
                "stream_child_outputs",
                "fuse_adjacent_children",
                "compile_whole_composition",
            ),
            admission_rule=_composition_execution_axis(),
        ),
        AxisDescriptor(
            "composition.child_evaluation",
            ("composition.child_evaluation",),
            ("selected_child_rows", "inline_child_lowering"),
        ),
        AxisDescriptor(
            "composition.validation",
            ("composition.validation",),
            ("validate_each_child", "validate_composed_output"),
        ),
        AxisDescriptor(
            "numeric.bf16_reduced_precision_reduction",
            ("numeric.bf16_reduced_precision_reduction",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "numeric.fp16_reduced_precision_reduction",
            ("numeric.fp16_reduced_precision_reduction",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "numeric.deterministic_algorithms",
            ("numeric.deterministic_algorithms",),
            ("false", "true"),
        ),
        AxisDescriptor(
            "numeric.loss_scaling",
            ("numeric.loss_scaling",),
            ("none", "static_scale_with_exact_unscale"),
            optional_settings_keys=(
                "numeric.loss_scale",
                "numeric.loss_unscale_degree",
            ),
            admission_rule=_loss_scaling_axis(),
        ),
    )


def standard_axis_registry(*, exclude: Sequence[str] = ()) -> AxisRegistry:
    """Return a registry populated with standard core axes."""
    registry = AxisRegistry()
    excluded = set(exclude)

    for axis in standard_axis_descriptors():
        if axis.name in excluded:
            continue

        registry.register(axis)

    return registry


def settings_product(
    family: str,
    axes: Mapping[str, Sequence[Any]],
    *,
    axis_registry: AxisRegistry | None = None,
    generator_id: str = "grid",
    generator_version: str = "0.0.1",
) -> tuple[Candidate, ...]:
    """Create candidates from an ordered grid of axis values.

    Returns:
        Candidate grid in deterministic order.
    """
    items = tuple(axes.items())
    candidates = []

    def build(index: int, settings: dict[str, Any], changed: tuple[str, ...]) -> None:
        if index == len(items):
            candidate_id = f"{family}:{len(candidates)}"
            candidates.append(
                Candidate(
                    family=family,
                    candidate_id=candidate_id,
                    settings=dict(settings),
                    changed_axes=changed,
                    generator_id=generator_id,
                    generator_version=generator_version,
                )
            )

            return

        axis_name, values = items[index]

        for value in values:
            changed_settings = _settings_for_axis_value(
                axis_name,
                value,
                axis_registry,
            )
            settings.update(changed_settings)
            build(index + 1, settings, (*changed, axis_name))

        for key in changed_settings:
            del settings[key]

    build(0, {}, ())

    return tuple(candidates)


def _settings_for_axis_value(
    axis_name: str,
    value: Any,
    axis_registry: AxisRegistry | None,
) -> dict[str, Any]:
    if axis_registry is None:
        return {axis_name: value}

    axis = axis_registry.axes.get(axis_name)

    if axis is None:
        message = f"candidate axis is unknown: {axis_name}"
        raise AdmissionError(message)

    if len(axis.settings_keys) == 1:
        return {axis.settings_keys[0]: value}

    if not isinstance(value, Mapping):
        message = f"multi-key axis value must be a mapping: {axis_name}"
        raise AdmissionError(message)

    if set(value) != set(axis.settings_keys):
        message = f"multi-key axis value keys differ from axis settings: {axis_name}"
        raise AdmissionError(message)

    return {key: value[key] for key in axis.settings_keys}
