"""Candidate axes, admission, and DAG helpers."""

import dataclasses
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vptune.axes.admission import (
    FORWARD_AD_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_forward_ad,
    admit_torch_func,
)
from vptune.core.data import PACKAGE_VERSION, Candidate, Family
from vptune.errors import AdmissionError

AdmissionRule = Callable[[Candidate], tuple[bool, str | None]]


@dataclasses.dataclass(frozen=True, slots=True)
class AxisTable:
    """Complete sweep axis table."""

    package_version: str
    axis_table_version: str
    axes: tuple["AxisDescriptor", ...]
    class_c_groups: Mapping[str, tuple[str, ...]]
    merge_rules: tuple[str, ...]

    def by_key(self) -> dict[str, "AxisDescriptor"]:
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


AxisManifest = AxisTable


ALL_OPERATOR_FAMILIES = (
    "gradient",
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "per_example_gradient",
    "metric",
    "sqrt_metric",
    "inverse_sqrt_metric",
    "metric_inner",
    "inverse_metric",
    "inverse_metric_inner",
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
    "per_example_gradient",
)
VECTOR_OPERATOR_FAMILIES = (
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "sqrt_metric",
    "inverse_sqrt_metric",
    "metric_inner",
    "composition",
    "inverse_metric",
    "inverse_metric_inner",
)
BOUND_OPERATOR_VECTOR_STEP_FAMILIES = (
    "jvp",
    "vjp",
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "metric",
    "sqrt_metric",
    "inverse_sqrt_metric",
    "inverse_metric",
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
DIRECT_LAYOUT_VALUES = (
    "parameter_tree",
    "flat_contiguous",
    "per_layer_flat",
    "per_block_flat",
)
DISTRIBUTED_LAYOUT_VALUES = ("per_shard", "dtensor")

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
FSDP_RESHARD_AFTER_FORWARD_DOMAIN = ("false_true_or_positive_integer_domain",)
COMPILE_BOUNDARY_VALUES = (
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
    "metric_inner_reduce",
    "metric_sqrt_multiply",
    "inverse_metric_solve",
    "inverse_metric_inner_reduce",
    "per_example_gradient",
    "bound_operator_vector_step",
    "composition_child",
    "whole_operator",
)
BACKEND_OPTION_KEYS = (
    "compile.options.epilogue_fusion",
    "compile.options.shape_padding",
    "compile.cuda_graphs",
)
COMPILE_REQUIRED_ENABLED_SETTINGS = (
    "compile.boundary",
    "compile.backend",
    "compile.mode",
    "compile.fullgraph",
    "compile.dynamic",
    "compile.compiled_autograd",
    *BACKEND_OPTION_KEYS,
    "compile.cache_state",
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
INTERNAL_ADMISSION_AXES = ("forward_ad_flags", "torch_func_admission")
OWNER_EXACT = {
    "teacher_outputs": "input_schedule",
    "autocast": "numeric_backend",
    "forward_ad_flags": "ad_lowering",
    "torch_func_admission": "ad_lowering",
    "attention.custom_kernel_id": "transformers_attention",
    "attention.mask_formatter_id": "transformers_attention",
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
    "per_example_gradient": "per_example_gradient",
    "metric": "metric",
    "metric_inner": "metric",
    "sqrt_metric": "metric",
    "inverse_metric": "inverse_metric",
    "inverse_metric_inner": "inverse_metric",
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
    "forward_ad_flags": "ad_lowering",
    "torch_func_admission": "ad_lowering",
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
    "per_example_gradient": "ad_lowering",
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
    "metric_inner": "metric_storage",
    "sqrt_metric": "metric_storage",
    "inverse_metric": "inverse_solve",
    "inverse_metric_inner": "inverse_solve",
}
OPERATOR_REFERENCE_CHECKS = {
    "gradient": (
        "direct_autograd_anchor",
        "finite_difference_directional",
    ),
    "jvp": (
        "jvp_anchor",
        "finite_difference_directional",
        "jvp_vjp_dot_identity",
    ),
    "vjp": (
        "vjp_anchor",
        "jvp_vjp_dot_identity",
    ),
    "hvp": (
        "reverse_over_reverse_anchor",
        "autograd_functional_anchor",
        "hvp_symmetry",
        "finite_difference_gradient_directional",
    ),
    "ggn": (
        "dense_jacobian_ggn_anchor",
        "jvp_hessian_vjp_cross_check",
        "loss_hessian_symmetry",
        "loss_hessian_psd",
    ),
    "fisher": (
        "explicit_score_outer_product_anchor",
        "dense_fisher_anchor",
    ),
    "sampled_fisher": (
        "fixed_sample_source_check",
        "explicit_sampled_score_outer_product_anchor",
        "dense_sampled_fisher_anchor",
    ),
    "empirical_fisher": (
        "per_example_gradient_loop_anchor",
        "dense_empirical_fisher_anchor",
    ),
    "per_example_gradient": (
        "per_example_gradient_loop_anchor",
        "empirical_fisher_outer_product_check",
    ),
    "metric": (
        "dense_metric_reference",
        "metric_symmetry_check",
        "metric_psd_check",
    ),
    "metric_inner": (
        "dense_metric_gram_reference",
        "metric_inner_diagonal_nonnegative_check",
    ),
    "sqrt_metric": (
        "dense_factor_check",
        "matrix_free_covariance_check",
    ),
    "inverse_metric": (
        "dense_inverse_reference",
        "inverse_residual_check",
    ),
    "inverse_metric_inner": (
        "dense_inverse_gram_reference",
        "inverse_inner_residual_check",
    ),
    "composition": (
        "child_anchor_checks",
        "dense_composed_output_check",
        "dependency_identity_equality",
    ),
}
SHARED_REFERENCE_CHECKS = {
    "attention": ("attention_backend_equality",),
    "batch": ("segmentation_invariance",),
    "chunk": ("segmentation_invariance",),
    "schedule": ("segmentation_invariance",),
    "input": ("input_representation_equality",),
    "teacher_outputs": ("teacher_output_equality",),
    "activation": ("recompute_or_offload_equality",),
    "checkpoint": ("checkpoint_recompute_equality",),
    "dtype": ("dtype_reference_agreement",),
    "numeric": ("numeric_error_bound_check",),
    "autocast": ("dtype_reference_agreement",),
    "fusion": ("fused_kernel_reference_agreement",),
    "layout": ("layout_roundtrip_reference",),
    "dtensor": ("distributed_logical_output_agreement",),
    "distributed": ("distributed_logical_output_agreement",),
    "fsdp": ("distributed_logical_output_agreement",),
    "tp": ("distributed_logical_output_agreement",),
    "sequence_parallel": ("distributed_logical_output_agreement",),
    "context_parallel": ("distributed_logical_output_agreement",),
    "comm": ("distributed_logical_output_agreement",),
}
FULL_SIZE_CHECK_PREFIXES = {
    "attention",
    "compile",
    "fusion",
    "dtensor",
    "distributed",
    "fsdp",
    "tp",
    "sequence_parallel",
    "context_parallel",
    "comm",
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
    "per_example_gradient": ("per_example_gradient",),
    "metric": ("metric",),
    "metric_inner": ("metric_inner",),
    "sqrt_metric": ("sqrt_metric", "inverse_sqrt_metric"),
    "inverse_metric": ("inverse_metric",),
    "inverse_metric_inner": ("inverse_metric_inner",),
    "composition": ("composition",),
}
AXIS_TABLE_DOMAIN_OVERRIDES = {
    "vectorization.batch_size": INTEGER_DOMAIN,
    "vectorization.vmap_chunk_size": INTEGER_DOMAIN,
    "vectorization.in_dims": DECLARED_DOMAIN,
    "batch.data_microbatch_size": INTEGER_DOMAIN,
    "batch.hvp_row_batch_size": INTEGER_DOMAIN,
    "batch.ggn_batch_size": INTEGER_DOMAIN,
    "batch.fisher_sample_batch_size": INTEGER_DOMAIN,
    "batch.empirical_example_batch_size": INTEGER_DOMAIN,
    "batch.per_example_block_size": INTEGER_DOMAIN,
    "chunk.token_block_size": INTEGER_DOMAIN,
    "chunk.sequence_position_block_size": INTEGER_DOMAIN,
    "chunk.class_block_size_with_exact_global_normalization": INTEGER_DOMAIN,
    "chunk.output_cotangent_block_size": INTEGER_DOMAIN,
    "chunk.parameter_block_size": INTEGER_DOMAIN,
    "chunk.layer_block_size": INTEGER_DOMAIN,
    "chunk.lm_head_weight_chunk_bytes": INTEGER_DOMAIN,
    "attention.custom_kernel_id": REGISTERED_DOMAIN,
    "attention.mask_formatter_id": REGISTERED_DOMAIN,
    "compile.backend": ("inductor", "registered_backend"),
    "distributed.mesh_shape": INTEGER_TUPLE_DOMAIN,
    "distributed.mesh_dim_names": DECLARED_DOMAIN,
    "fsdp.reshard_after_forward": FSDP_RESHARD_AFTER_FORWARD_DOMAIN,
    "fsdp.ignored_params": DECLARED_DOMAIN,
    "fsdp.dp_mesh_dims": DECLARED_DOMAIN,
    "inverse_metric.iteration_budget": INTEGER_DOMAIN,
    "sqrt_metric.lanczos_iterations": INTEGER_DOMAIN,
    "tp.plan": REGISTERED_DOMAIN,
    "tp.prepare_module_input": DECLARED_DOMAIN,
    "tp.prepare_module_output": DECLARED_DOMAIN,
    "sequence_parallel.norm_modules": DECLARED_DOMAIN,
    "context_parallel.sequence_dim": DECLARED_DOMAIN,
    "comm.collective_bucket_size": INTEGER_DOMAIN,
}


def axis_table() -> AxisTable:
    """Return the complete sweep axis table.

    Returns:
        Complete axis table for candidate generation and admission.
    """
    axes = _axis_table_axes()
    groups = {}

    for axis in axes:
        groups.setdefault(axis.class_c_group, []).append(axis.axis_key)

    class_c_groups = {name: tuple(keys) for name, keys in sorted(groups.items())}

    return AxisTable(
        package_version=PACKAGE_VERSION,
        axis_table_version="1",
        axes=axes,
        class_c_groups=class_c_groups,
        merge_rules=AXIS_TABLE_MERGE_RULES,
    )


def axis_manifest() -> AxisManifest:
    """Return the complete sweep axis table.

    Returns:
        Complete axis table for candidate generation and admission.
    """
    return axis_table()


def _axis_table_axes() -> tuple["AxisDescriptor", ...]:
    standard_axes = tuple(
        _axis_table_axis_from_descriptor(axis)
        for axis in standard_axis_descriptors()
        if axis.axis_key not in INTERNAL_ADMISSION_AXES
    )
    adapter_axes = tuple(
        _axis_table_axis_from_descriptor(axis)
        for axis in _axis_table_adapter_axis_descriptors()
    )
    seen = set()
    axes = []

    for axis in (*standard_axes, *adapter_axes):
        if axis.axis_key in seen:
            message = f"axis table axis is registered twice: {axis.axis_key}"
            raise AdmissionError(message)

        seen.add(axis.axis_key)
        axes.append(axis)

    return tuple(axes)


def _axis_table_axis_from_descriptor(axis: "AxisDescriptor") -> "AxisDescriptor":
    axis_key = axis.axis_key
    class_c_group = _class_c_group(axis_key)

    return AxisDescriptor(
        name=axis_key,
        settings_keys=axis.settings_keys,
        allowed_values=_axis_table_descriptor_domain(axis),
        optional_settings_keys=axis.optional_settings_keys,
        owner_id=_owner_id(axis_key),
        value_owner_ids=_value_owner_ids(axis_key),
        operators=_operators_for_axis(axis_key),
        class_a=_class_a(axis_key),
        class_b=_class_b(axis_key),
        class_c_group=class_c_group,
        merge_rules=_axis_merge_rules(axis_key),
        adapter_id=_adapter_id(axis_key),
        adapter_version=axis.adapter_version,
        admission_rule=axis.admission_rule,
        admission_rule_id=_axis_admission_rule_id(axis_key, axis),
        lowering_rule_id=_axis_lowering_rule_id(axis_key),
        alias_normalization_rule=_axis_alias_normalization_rule(axis_key),
        required_reference_checks=_axis_required_reference_checks(axis_key),
        required_full_size_checks=_axis_required_full_size_checks(axis_key),
        admission_settings_keys=_axis_admission_settings_keys(axis_key, axis),
        settings_keys_written=axis.settings_keys,
        identity=axis.identity,
    )


def _axis_table_descriptor_domain(axis: "AxisDescriptor") -> tuple[Any, ...]:
    override = AXIS_TABLE_DOMAIN_OVERRIDES.get(axis.axis_key)

    if override is not None:
        return override

    if axis.admission_rule is not None:
        return axis.allowed_values

    if not axis.allowed_values:
        message = f"axis table domain is missing for descriptor: {axis.axis_key}"
        raise AdmissionError(message)

    return axis.allowed_values


def _axis_table_adapter_axis_descriptors() -> tuple["AxisDescriptor", ...]:
    attention_module = importlib.import_module("vptune.engine.attention")
    distributed_module = importlib.import_module("vptune.adapters.distributed")

    core_attention_axes = tuple(
        axis
        for axis in attention_module.core_attention_axis_descriptors()
        if axis.axis_key
        in {"attention.sdpa_kernel", "attention.partition", "attention.padding"}
    )

    return (
        _axis_table_attention_frontend_descriptor(),
        *_axis_table_transformers_optional_attention_descriptors(),
        *core_attention_axes,
        *distributed_module.distributed_manifest_axis_descriptors(),
    )


def _axis_table_attention_frontend_descriptor() -> "AxisDescriptor":
    attention_module = importlib.import_module("vptune.engine.attention")
    transformers_module = importlib.import_module("vptune.adapters.transformers")

    core_axis = attention_module.core_attention_axis()
    transformers_axis = transformers_module.transformers_manifest_attention_axis()

    return AxisDescriptor(
        "attention.frontend",
        ("attention.frontend",),
        _attention_frontend_values(),
        optional_settings_keys=tuple(
            dict.fromkeys((
                *core_axis.optional_settings_keys,
                *transformers_axis.optional_settings_keys,
            ))
        ),
        adapter_id="vptune.attention_or_adapter",
        adapter_version=PACKAGE_VERSION,
        admission_rule=_attention_frontend_manifest_axis(core_axis, transformers_axis),
        identity={
            "core": core_axis.signature(),
            "transformers": transformers_axis.signature(),
        },
    )


def _attention_frontend_manifest_axis(
    core_axis: "AxisDescriptor",
    transformers_axis: "AxisDescriptor",
) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings.get("attention.frontend")

        if value in core_axis.allowed_values:
            return core_axis.admit(candidate)

        if value in transformers_axis.allowed_values:
            return transformers_axis.admit(candidate)

        return False, f"attention.frontend is unsupported: {value}"

    return admit


def _axis_table_transformers_optional_attention_descriptors() -> tuple[
    "AxisDescriptor",
    ...,
]:
    transformers_module = importlib.import_module("vptune.adapters.transformers")

    transformers_axis = transformers_module.transformers_manifest_attention_axis()

    return tuple(
        AxisDescriptor(
            key,
            (key,),
            REGISTERED_DOMAIN,
            adapter_id="vptune.adapters.transformers",
            adapter_version=transformers_axis.adapter_version,
            admission_rule=_registered_id_axis(key),
            identity=transformers_axis.identity,
        )
        for key in ("attention.custom_kernel_id", "attention.mask_formatter_id")
        if key in transformers_axis.optional_settings_keys
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


def _value_owner_ids(axis_key: str) -> Mapping[Any, str]:
    if axis_key != "attention.frontend":
        return {}

    return {
        value: _attention_frontend_value_owner(value)
        for value in _attention_frontend_values()
    }


def _attention_frontend_value_owner(value: str) -> str:
    if value in _core_attention_frontend_values():
        return "attention_execution"

    return "vptune.adapters.transformers"


def _attention_frontend_values() -> tuple[str, ...]:
    transformers_module = importlib.import_module("vptune.adapters.transformers")

    return (
        *transformers_module.TRANSFORMERS_ATTENTION_FRONTENDS,
        *_core_attention_frontend_values(),
    )


def _core_attention_frontend_values() -> tuple[str, ...]:
    attention_module = importlib.import_module("vptune.engine.attention")

    return attention_module.CORE_ATTENTION_FRONTENDS


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

    if axis_key in {
        "inverse_metric.solve_path",
        "inverse_metric.preconditioner",
        "inverse_metric_inner.reduction_path",
    }:
        rules.append(FACTORIZED_INVERSE_MERGE_RULE)

    return tuple(rules)


def _adapter_id(axis_key: str) -> str:
    prefix = axis_key.split(".", 1)[0]

    if axis_key in {
        "attention.custom_kernel_id",
        "attention.mask_formatter_id",
    }:
        return "vptune.adapters.transformers"

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


def _axis_admission_rule_id(axis_key: str, axis: "AxisDescriptor") -> str:
    if axis.admission_rule is None:
        return f"allowed_values:{axis_key}"

    return f"callable:{axis_key}"


def _axis_lowering_rule_id(axis_key: str) -> str:
    adapter_id = _adapter_id(axis_key)

    if adapter_id:
        return f"{adapter_id}:{axis_key}"

    return f"vptune.standard_runtime:{_owner_id(axis_key)}"


def _axis_alias_normalization_rule(axis_key: str) -> str:
    if axis_key.startswith("compile."):
        return "compile_alias_normalization"

    if axis_key == "attention.sdpa_kernel":
        return "sdpa_priority_list_normalization"

    return "none"


def _axis_required_reference_checks(axis_key: str) -> tuple[str, ...]:
    prefix = axis_key.split(".", 1)[0]
    operator_checks = OPERATOR_REFERENCE_CHECKS.get(prefix)

    if operator_checks is not None:
        return operator_checks

    shared_checks = SHARED_REFERENCE_CHECKS.get(axis_key)

    if shared_checks is not None:
        return shared_checks

    shared_checks = SHARED_REFERENCE_CHECKS.get(prefix)

    if shared_checks is not None:
        return shared_checks

    return ()


def _axis_required_full_size_checks(axis_key: str) -> tuple[str, ...]:
    prefix = axis_key.split(".", 1)[0]

    if prefix in FULL_SIZE_CHECK_PREFIXES:
        return ("full_size_agreement",)

    return ()


def _axis_admission_settings_keys(
    axis_key: str,
    axis: "AxisDescriptor",
) -> tuple[str, ...]:
    keys = dict.fromkeys((*axis.settings_keys, *axis.optional_settings_keys))

    if axis_key == "compile.enabled":
        keys.update(dict.fromkeys(COMPILE_REQUIRED_ENABLED_SETTINGS))

    if axis_key == "attention.frontend":
        keys.update(dict.fromkeys(("attention.sdpa_kernel",)))

    return tuple(keys)


def _positive_integer_value_error(axis_key: str, value: Any) -> str | None:
    if not _positive_int_value(value):
        return f"candidate axis must be a positive integer: {axis_key}"

    return None


def _positive_integer_tuple_value_error(axis_key: str, value: Any) -> str | None:
    if not isinstance(value, tuple) or not value:
        return f"candidate axis must be a positive integer tuple: {axis_key}"

    for item in value:
        if not _positive_int_value(item):
            return f"candidate axis must be a positive integer tuple: {axis_key}"

    return None


def _positive_int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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


def _fsdp_reshard_after_forward_value_error(
    axis_key: str,
    value: Any,
) -> str | None:
    if value in {"false", "true"}:
        return None

    if not _positive_int_value(value):
        return f"candidate axis must be false, true, or a positive integer: {axis_key}"

    return None


def _compile_rule_error(
    settings: Mapping[str, Any],
    _: Mapping[str, Any],
) -> str | None:
    if settings.get("compile.enabled") == "true":
        for key in COMPILE_REQUIRED_ENABLED_SETTINGS:
            if key not in settings:
                return f"compile.enabled=true requires {key}"

    if settings.get("compile.enabled") == "false":
        if "compile.boundary" in settings:
            return "compile.enabled=false forbids compile.boundary"

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


BASELINE_ATTENTION_FRONTEND_VALUES = (
    "transformers_eager",
    "transformers_sdpa",
    "pytorch_sdpa_direct",
    "patched_eager",
)


def attention_frontend_requires_full_size_agreement(frontend: object) -> bool:
    """Return whether an attention frontend needs full-size agreement."""
    if not isinstance(frontend, str):
        return True

    return frontend not in BASELINE_ATTENTION_FRONTEND_VALUES


FORWARD_AD_TRANSFORM_PATHS = ("torch_func_jvp", "jvp_grad")
TORCH_FUNC_AXIS_EXCLUDED_FIELDS = {
    *FORWARD_AD_FIELDS,
    "vectorization.randomness",
}
TORCH_FUNC_AXIS_FIELDS = tuple(
    field for field in TORCH_FUNC_FIELDS if field not in TORCH_FUNC_AXIS_EXCLUDED_FIELDS
)
VMAP_TRANSFORM_PATHS = ("per_example_gradient_vmap",)
TORCH_FUNC_PATH_ADMISSION = {
    "gradient.path": {
        "torch_func_grad": "torch_func_vjp",
        "torch_func_grad_and_value": "torch_func_vjp",
    },
    "jvp.path": {
        "torch_func_jvp": "torch_func_jvp",
        "torch_func_linearize": "torch_func_jvp",
    },
    "vjp.path": {"torch_func_vjp": "torch_func_vjp"},
    "hvp.path": {
        "jvp_grad": "jvp_grad",
        "linearize_grad": "jvp_grad",
    },
    "ggn.jvp_path": {
        "torch_func_jvp": "torch_func_jvp",
        "torch_func_linearize": "torch_func_jvp",
    },
    "ggn.vjp_path": {"torch_func_vjp": "torch_func_vjp"},
    "fisher.score_grad_path": {
        "torch_func_grad": "torch_func_vjp",
        "vmap_grad": "per_example_gradient_vmap",
    },
    "sampled_fisher.score_grad_path": {
        "torch_func_grad": "torch_func_vjp",
        "vmap_grad": "per_example_gradient_vmap",
    },
    "empirical_fisher.grad_path": {
        "torch_func_grad": "torch_func_vjp",
        "vmap_grad": "per_example_gradient_vmap",
    },
    "per_example_gradient.grad_path": {
        "torch_func_grad": "torch_func_vjp",
        "vmap_grad": "per_example_gradient_vmap",
    },
}
FORWARD_AD_PATH_ADMISSION = {
    "jvp.path": ("forward_ad_dual",),
    "hvp.path": ("forward_ad_dual",),
    "ggn.jvp_path": ("forward_ad_dual",),
}
TORCH_FUNC_ADMISSION_PATH_SETTINGS = tuple(
    (key, value)
    for key, value_map in TORCH_FUNC_PATH_ADMISSION.items()
    for value in value_map
)
PER_EXAMPLE_SCORE_PATH_KEYS = (
    "fisher.score_grad_path",
    "sampled_fisher.score_grad_path",
    "empirical_fisher.grad_path",
)
MANUAL_PER_EXAMPLE_BATCH_SIZE_KEYS = (
    (("empirical_fisher.grad_path",), "batch.empirical_example_batch_size"),
    (
        ("fisher.score_grad_path", "sampled_fisher.score_grad_path"),
        "batch.fisher_sample_batch_size",
    ),
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
INVERSE_METRIC_VECTOR_LOOP_PATHS = (
    "dense_solve",
    "cholesky_solve",
    "eigh_solve",
    "svd_solve",
    "conjugate_gradient",
    "factorized_solve",
    "blockwise_solve",
    "woodbury_low_rank_solve",
)
METRIC_INNER_VECTOR_LOOP_PATHS = (
    "multiply_then_reduce",
    "factored_gram",
    "sqrt_apply_reduce",
)
INVERSE_METRIC_INNER_VECTOR_LOOP_PATHS = (
    "solve_then_reduce",
    "factored_gram",
    "sqrt_apply_reduce",
)
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
COMPOSITION_VECTOR_EXECUTIONS = (
    "materialize_each_child",
    "stream_child_outputs",
    "fuse_adjacent_children",
    "compile_whole_composition",
)
METRIC_INNER_VECTOR_PATH_SETTINGS = (
    ("metric_inner.reduction_path", METRIC_INNER_VECTOR_LOOP_PATHS),
    ("inverse_metric_inner.reduction_path", INVERSE_METRIC_INNER_VECTOR_LOOP_PATHS),
)
FISHER_FAMILY_VECTOR_PATH_SETTINGS = (
    ("fisher.accumulation", FISHER_VECTOR_ACCUMULATIONS),
    ("sampled_fisher.accumulation", SAMPLED_FISHER_VECTOR_ACCUMULATIONS),
    ("empirical_fisher.grad_path", EMPIRICAL_FISHER_VECTOR_GRAD_PATHS),
    ("empirical_fisher.accumulation", EMPIRICAL_FISHER_VECTOR_ACCUMULATIONS),
)
VECTOR_LOOP_PATH_SETTINGS = (
    ("hvp.path", HVP_VECTOR_LOOP_PATHS),
    ("jvp.path", JVP_VECTOR_LOOP_PATHS),
    ("vjp.path", VJP_VECTOR_LOOP_PATHS),
    ("inverse_metric.solve_path", INVERSE_METRIC_VECTOR_LOOP_PATHS),
    *METRIC_INNER_VECTOR_PATH_SETTINGS,
    *FISHER_FAMILY_VECTOR_PATH_SETTINGS,
)
VECTOR_VMAP_PATH_SETTINGS = (
    ("hvp.path", HVP_VECTOR_VMAP_PATHS),
    ("jvp.path", JVP_VECTOR_VMAP_PATHS),
    ("vjp.path", VJP_VECTOR_VMAP_PATHS),
    *METRIC_INNER_VECTOR_PATH_SETTINGS,
    *FISHER_FAMILY_VECTOR_PATH_SETTINGS,
)
MATMUL_PRECISION_VALUES = ("highest", "high", "medium")
SPEC_DTYPE_VALUES = ("fp32", "bf16", "fp16")
SPEC_STORAGE_COMPUTE_DTYPE_VALUES = (*SPEC_DTYPE_VALUES, "fp8_when_supported")


@dataclasses.dataclass(frozen=True, slots=True)
class AxisDescriptor:
    """One tunable axis registered by core or an adapter."""

    name: str
    settings_keys: tuple[str, ...]
    allowed_values: tuple[Any, ...]
    optional_settings_keys: tuple[str, ...] = ()
    owner_id: str = "core"
    value_owner_ids: Mapping[Any, str] = dataclasses.field(default_factory=dict)
    operators: tuple[str, ...] = ()
    class_a: str = ""
    class_b: str = ""
    class_c_group: str = ""
    merge_rules: tuple[str, ...] = ()
    adapter_id: str = "core"
    adapter_version: str = dataclasses.field(default_factory=lambda: PACKAGE_VERSION)
    admission_rule: AdmissionRule | None = None
    admission_rule_id: str = ""
    lowering_rule_id: str = ""
    alias_normalization_rule: str = "none"
    required_reference_checks: tuple[str, ...] = ()
    required_full_size_checks: tuple[str, ...] = ()
    admission_settings_keys: tuple[str, ...] = ()
    settings_keys_written: tuple[str, ...] = ()
    identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def axis_key(self) -> str:
        """Return the axis key."""
        return self.name

    @property
    def value_domain(self) -> tuple[Any, ...]:
        """Return the allowed value domain."""
        return self.allowed_values

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
            "axis_key": self.axis_key,
            "settings_keys": self.settings_keys,
            "optional_settings_keys": self.optional_settings_keys,
            "allowed_values": self.allowed_values,
            "value_domain": self.value_domain,
            "owner_id": self.owner_id,
            "value_owner_ids": dict(self.value_owner_ids),
            "operators": self.operators,
            "class_a": self.class_a,
            "class_b": self.class_b,
            "class_c_group": self.class_c_group,
            "merge_rules": self.merge_rules,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "has_admission_rule": self.admission_rule is not None,
            "admission_rule_id": self.admission_rule_id,
            "lowering_rule_id": self.lowering_rule_id,
            "alias_normalization_rule": self.alias_normalization_rule,
            "required_reference_checks": self.required_reference_checks,
            "required_full_size_checks": self.required_full_size_checks,
            "admission_settings_keys": self.admission_settings_keys,
            "settings_keys_written": self.settings_keys_written,
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

            optional_owners = self.optional_owners.get(key)

            if optional_owners is not None:
                message = f"setting key has primary and optional axis owners: {key}"
                raise AdmissionError(message)

        self.axes[axis.name] = axis

        for key in axis.settings_keys:
            self.owners[key] = axis.name

        for key in axis.optional_settings_keys:
            owner = self.owners.get(key)

            if owner is not None:
                message = f"setting key has primary and optional axis owners: {key}"
                raise AdmissionError(message)

            owners = self.optional_owners.get(key, ())

            if owners and axis.name not in owners:
                message = f"setting key has multiple optional axis owners: {key}"
                raise AdmissionError(message)

            if not owners:
                self.optional_owners[key] = (axis.name,)

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

    if _is_symbolic_axis_domain(axis.allowed_values):
        return _symbolic_axis_domain_error(axis.axis_key, axis.allowed_values, value)

    if not axis.allowed_values or any(
        value == allowed for allowed in axis.allowed_values
    ):
        return None

    return f"candidate axis value is not allowed: {axis.name}"


def _symbolic_axis_domain_error(
    axis_key: str,
    domain: tuple[Any, ...],
    value: Any,
) -> str | None:
    validators = {
        INTEGER_DOMAIN: _positive_integer_value_error,
        INTEGER_TUPLE_DOMAIN: _positive_integer_tuple_value_error,
        POSITIVE_FLOAT_DOMAIN: _positive_float_value_error,
        DECLARED_DOMAIN: _declared_value_error,
        REGISTERED_DOMAIN: _registered_value_error,
        FSDP_RESHARD_AFTER_FORWARD_DOMAIN: (_fsdp_reshard_after_forward_value_error),
    }
    validate = validators.get(domain)

    if validate is None:
        return None

    return validate(axis_key, value)


def _is_symbolic_axis_domain(domain: tuple[Any, ...]) -> bool:
    return any(
        domain == symbolic_domain
        for symbolic_domain in (
            INTEGER_DOMAIN,
            INTEGER_TUPLE_DOMAIN,
            POSITIVE_FLOAT_DOMAIN,
            DECLARED_DOMAIN,
            REGISTERED_DOMAIN,
            FSDP_RESHARD_AFTER_FORWARD_DOMAIN,
        )
    )


def _compile_backend_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["compile.backend"]
        compile_error = _compile_rule_error(candidate.settings, {})

        if compile_error is not None:
            return False, compile_error

        if not isinstance(value, str):
            return False, "compile.backend must be a string"

        if value == "registered_backend":
            return False, "compile.backend requires a concrete PyTorch backend id"

        if value == "inductor" or value in set(torch.compiler.list_backends()):
            return True, None

        return False, f"compile.backend is not registered with PyTorch: {value}"

    return admit


def _compile_setting_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _compile_rule_error(candidate.settings, {})

        return (True, None) if error is None else (False, error)

    return admit


def _compile_boundary_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        settings = candidate.settings
        error = _compile_rule_error(settings, {})

        if error is not None:
            return False, error

        if settings.get("compile.enabled") != "true":
            return False, "compile.boundary requires compile.enabled=true"

        boundary = settings["compile.boundary"]

        if _adapter_compile_boundary_supported(boundary, settings):
            return True, None

        operator_kind = _compile_boundary_operator_kind(settings)

        if operator_kind is None:
            return False, "compile.boundary requires one operator path setting"

        if _compile_boundary_supported(operator_kind, boundary, settings):
            return True, None

        return False, f"compile.boundary={boundary} is not lowered for {operator_kind}"

    return admit


def _adapter_compile_boundary_supported(
    boundary: Any,
    settings: Mapping[str, Any],
) -> bool:
    if boundary == "attention_module" and "attention.frontend" in settings:
        return True

    return boundary == "transformer_block" and _has_transformers_adapter_setting(
        settings
    )


def _has_transformers_adapter_setting(settings: Mapping[str, Any]) -> bool:
    frontend = settings.get("attention.frontend")

    if isinstance(frontend, str) and (
        frontend.startswith("transformers_")
        or frontend == "registered_transformers_attention"
    ):
        return True

    return any(
        key in settings
        for key in (
            "module_mode",
            "dropout_p",
            "use_cache",
            "output_attentions",
        )
    )


def _memory_output_axis(key: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[key]

        if value == "retain":
            return True, None

        if value == "recompute":
            if _memory_output_recompute_has_lowering(key, candidate.settings):
                return True, None

            return False, f"{key}=recompute requires matching recompute settings"

        return False, f"{key} is unsupported: {value}"

    return admit


def _memory_output_recompute_has_lowering(
    key: str,
    settings: Mapping[str, Any],
) -> bool:
    if key == "memory.primal_outputs":
        return (
            settings.get("hvp.path") == "reverse_over_reverse"
            and settings.get("hvp.primal_reuse") == "recompute_primal"
        )

    if key == "memory.jvp_outputs":
        return settings.get("ggn.jvp_reuse") == "recompute_jvp"

    if key == "memory.output_cotangents":
        return settings.get("ggn.cotangent_reuse") == "recompute_output_cotangent"

    return False


def _memory_intermediate_residency_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["memory.intermediate_residency"]

        if value not in {"gpu", "cpu_staged", "cpu_pinned"}:
            return False, f"memory.intermediate_residency is unsupported: {value}"

        operator_kind = _compile_boundary_operator_kind(candidate.settings)

        if operator_kind == "ggnvp":
            return True, None

        if operator_kind == "composition":
            if (
                candidate.settings.get("composition.execution")
                == "fuse_adjacent_children"
            ):
                return (
                    False,
                    (
                        "memory.intermediate_residency requires visible composition "
                        "child boundaries"
                    ),
                )

            return True, None

        return (
            False,
            "memory.intermediate_residency requires named intermediate boundaries",
        )

    return admit


def _compile_boundary_operator_kind(settings: Mapping[str, Any]) -> str | None:
    operator_checks = (
        ("gradient", ("gradient.path",)),
        ("jvp", ("jvp.path",)),
        ("vjp", ("vjp.path",)),
        ("hvp", ("hvp.path",)),
        ("ggnvp", ("ggn.jvp_path", "ggn.loss_hessian_kernel")),
        ("fisher_vp", ("fisher.accumulation",)),
        ("sampled_fisher_vp", ("sampled_fisher.accumulation",)),
        (
            "empirical_fisher_vp",
            ("empirical_fisher.grad_path", "empirical_fisher.accumulation"),
        ),
        ("per_example_gradient", ("per_example_gradient.grad_path",)),
        ("metric_inner", ("metric_inner.reduction_path",)),
        ("inverse_metric_inner", ("inverse_metric_inner.reduction_path",)),
        ("sqrt_metric", ("sqrt_metric.factor_path",)),
        ("inverse_metric", ("inverse_metric.solve_path",)),
        ("metric", ("metric.multiply_path",)),
        ("composition", ("composition.execution",)),
    )
    compound_checks = (
        (
            "metric_inner",
            ("metric_inner.reduction_path",),
            ("metric.multiply_path", "sqrt_metric.factor_path"),
        ),
        (
            "inverse_metric_inner",
            ("inverse_metric_inner.reduction_path",),
            (
                "inverse_metric.solve_path",
                "metric.multiply_path",
                "sqrt_metric.factor_path",
            ),
        ),
        (
            "inverse_metric",
            ("inverse_metric.solve_path",),
            ("metric.multiply_path",),
        ),
    )

    for operator, keys, dependency_keys in compound_checks:
        if not any(key in settings for key in keys):
            continue

        if _has_other_operator_path_setting(
            settings,
            operator_checks,
            (*keys, *dependency_keys),
        ):
            return None

        return operator

    operators = tuple(
        operator
        for operator, keys in operator_checks
        if any(key in settings for key in keys)
    )

    if len(operators) == 1:
        return operators[0]

    return None


def _has_other_operator_path_setting(
    settings: Mapping[str, Any],
    operator_checks: Sequence[tuple[str, tuple[str, ...]]],
    allowed_keys: tuple[str, ...],
) -> bool:
    for _, keys in operator_checks:
        for key in keys:
            if key in settings and key not in allowed_keys:
                return True

    return False


def _compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    direct = _direct_compile_boundary_supported(operator_kind, boundary, settings)

    if direct is not None:
        return direct

    if operator_kind == "hvp":
        return _hvp_compile_boundary_supported(boundary, settings)

    if operator_kind == "ggnvp":
        return _ggn_compile_boundary_supported(boundary, settings)

    if operator_kind in {
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
        "per_example_gradient",
    }:
        return _score_matrix_compile_boundary_supported(
            operator_kind,
            boundary,
            settings,
        )

    boundaries = {
        "gradient": "gradient_closure",
        "jvp": "jvp_closure",
        "vjp": "vjp_closure",
        "metric": "metric_multiply",
        "sqrt_metric": "metric_sqrt_multiply",
        "inverse_sqrt_metric": "metric_sqrt_multiply",
        "metric_inner": "metric_inner_reduce",
        "inverse_metric": "inverse_metric_solve",
        "inverse_metric_inner": "inverse_metric_inner_reduce",
        "composition": "composition_child",
    }

    return boundaries.get(operator_kind) == boundary


def _direct_compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool | None:
    if boundary == "whole_operator":
        return True

    if boundary == "model_forward":
        return settings.get("call.path") == "stateful_module"

    if boundary == "loss_closure":
        return operator_kind in {"gradient", "hvp"}

    if boundary == "bound_operator_vector_step":
        return operator_kind in BOUND_OPERATOR_VECTOR_STEP_FAMILIES

    return None


def _hvp_compile_boundary_supported(
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    vectorized = settings.get("vectorization.mode") in {"single_loop", "vmap"}

    if boundary == "hvp_single_vector":
        return not vectorized

    if boundary == "hvp_batched_vectors":
        return vectorized

    return False


def _ggn_compile_boundary_supported(
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    if boundary == "ggn_full_product":
        return True

    if settings.get("vectorization.mode") in {"single_loop", "manual_batch", "vmap"}:
        return False

    if boundary in {"ggn_jvp", "ggn_loss_hessian_product"}:
        return settings.get("ggn.jvp_path") in {
            "torch_func_jvp",
            "forward_ad_dual",
            "torch_func_linearize",
        }

    if boundary != "ggn_vjp":
        return False

    return settings.get("ggn.vjp_path") in {
        "torch_func_vjp",
        "autograd_grad_outputs",
    }


def _score_matrix_compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    if operator_kind == "fisher_vp":
        return boundary == "fisher_score_grad" and settings.get(
            "fisher.score_grad_path"
        ) in {
            "torch_autograd_grad_loop",
            "torch_func_grad",
            "vmap_grad",
            "backward_materialized_grad",
        }

    if operator_kind == "sampled_fisher_vp":
        return boundary == "sampled_fisher_score_grad" and settings.get(
            "sampled_fisher.score_grad_path"
        ) in {
            "torch_autograd_grad_loop",
            "torch_func_grad",
            "vmap_grad",
            "backward_materialized_grad",
        }

    if operator_kind == "empirical_fisher_vp":
        return boundary == "empirical_fisher_example_grad" and settings.get(
            "empirical_fisher.grad_path"
        ) in {
            "torch_autograd_grad_loop",
            "torch_func_grad",
            "vmap_grad",
            "backward_materialized_grad",
        }

    return (
        boundary == "per_example_gradient"
        and settings.get("per_example_gradient.grad_path")
        in {
            "torch_autograd_grad_loop",
            "torch_func_grad",
            "vmap_grad",
            "backward_materialized_grad",
        }
        and settings.get("per_example_gradient.accumulation") == "stacked_leading_axis"
    )


def _positive_int_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            error = _positive_int_error(candidate.settings, key)

            if error is not None:
                return False, error

        return True, None

    return admit


def _registered_id_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            value = candidate.settings[key]

            if not isinstance(value, str) or not value:
                return False, f"{key} must be a registered id"

        return True, None

    return admit


def _fsdp_reshard_after_forward_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        key = "fsdp.reshard_after_forward"
        error = _fsdp_reshard_after_forward_value_error(
            key,
            candidate.settings[key],
        )

        if error is not None:
            return False, error

        return True, None

    return admit


def _positive_int_error(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings[key]

    if not _positive_int_value(value):
        return f"candidate axis must be a positive integer: {key}"

    return None


def _bool_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            if not isinstance(candidate.settings[key], bool):
                return False, f"candidate axis must be boolean: {key}"

        return True, None

    return admit


def _activation_offload_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if candidate.settings["activation.offload"] != "custom_saved_tensor_hooks":
            return True, None

        pack_hook = candidate.settings.get("activation.pack_hook")
        unpack_hook = candidate.settings.get("activation.unpack_hook")

        if not isinstance(pack_hook, str) or not pack_hook:
            return False, "activation.pack_hook must be a declared hook id"

        if not isinstance(unpack_hook, str) or not unpack_hook:
            return False, "activation.unpack_hook must be a declared hook id"

        return True, None

    return admit


def _checkpoint_context_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if candidate.settings["checkpoint.context_fn"] != "declared_context_pair":
            return True, None

        context_id = candidate.settings.get("checkpoint.context_fn_callable")

        if not isinstance(context_id, str) or not context_id:
            return False, "checkpoint.context_fn_callable must be a declared context id"

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


def _path_admission_axis(key: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[key]
        torch_func_path = TORCH_FUNC_PATH_ADMISSION.get(key, {}).get(value)

        if torch_func_path is not None:
            return _admit_torch_func_path(torch_func_path, candidate.settings)

        if value in FORWARD_AD_PATH_ADMISSION.get(key, ()):
            return _admit_forward_ad_path(candidate.settings)

        return True, None

    return admit


def _storage_compute_dtype_axis(key: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[key]

        if value != "fp8_when_supported":
            return True, None

        if hasattr(torch, "float8_e4m3fn"):
            return True, None

        return False, f"{key}=fp8_when_supported requires PyTorch FP8 dtype support"

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


def _inverse_metric_multi_rhs_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        settings = candidate.settings
        value = settings["inverse_metric.multi_rhs"]

        if value == "single_column":
            return True, None

        if value != "block":
            return False, f"inverse_metric.multi_rhs is unsupported: {value}"

        if settings.get("vectorization.mode") not in {"single_loop", "manual_batch"}:
            return False, "inverse_metric.multi_rhs=block requires stacked vectors"

        return True, None

    return admit


def _inverse_metric_preconditioner_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["inverse_metric.preconditioner"]

        if value == "matrix_free":
            product = candidate.settings.get("inverse_metric.preconditioner_product")

            if isinstance(product, str) and product:
                return True, None

            return False, (
                "inverse_metric.preconditioner=matrix_free requires "
                "inverse_metric.preconditioner_product"
            )

        if "inverse_metric.preconditioner_product" in candidate.settings:
            return False, (
                "inverse_metric.preconditioner_product applies only to "
                "matrix_free preconditioner"
            )

        if value in {"none", "diagonal", "block_diagonal", "factorized_metric"}:
            return True, None

        return False, f"inverse_metric.preconditioner is unsupported: {value}"

    return admit


def _layout_tree_axis(key: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[key]

        if value in DIRECT_LAYOUT_VALUES:
            return True, None

        if value in DISTRIBUTED_LAYOUT_VALUES:
            if _has_distributed_adapter_setting(candidate.settings):
                return True, None

            return False, f"{key}={value} requires distributed adapter ownership"

        return False, f"{key} is unsupported: {value}"

    return admit


def _has_distributed_adapter_setting(settings: Mapping[str, Any]) -> bool:
    prefixes = (
        "distributed.",
        "dtensor.",
        "fsdp.",
        "tp.",
        "sequence_parallel.",
        "context_parallel.",
        "comm.",
    )

    return any(key.startswith(prefixes) for key in settings)


def _sqrt_metric_factor_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["sqrt_metric.factor_path"]

        if value != "matrix_free_lanczos":
            if "sqrt_metric.lanczos_iterations" in candidate.settings:
                return (
                    False,
                    "sqrt_metric.lanczos_iterations requires matrix_free_lanczos",
                )

            return True, None

        if "sqrt_metric.lanczos_iterations" not in candidate.settings:
            return (
                False,
                "matrix_free_lanczos requires sqrt_metric.lanczos_iterations",
            )

        error = _positive_int_error(
            candidate.settings, "sqrt_metric.lanczos_iterations"
        )

        return (True, None) if error is None else (False, error)

    return admit


def _sqrt_metric_lanczos_iterations_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if candidate.settings.get("sqrt_metric.factor_path") != "matrix_free_lanczos":
            return (
                False,
                "sqrt_metric.lanczos_iterations requires matrix_free_lanczos",
            )

        error = _positive_int_error(
            candidate.settings, "sqrt_metric.lanczos_iterations"
        )

        return (True, None) if error is None else (False, error)

    return admit


def _metric_inner_reduction_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _metric_inner_reduction_path_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _inverse_metric_inner_reduction_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _inverse_metric_inner_reduction_path_error(candidate.settings)

        return (True, None) if error is None else (False, error)

    return admit


def _metric_inner_reduction_path_error(settings: Mapping[str, Any]) -> str | None:
    return _inner_reduction_path_error(
        settings,
        "metric_inner.reduction_path",
        "metric.multiply_path",
        "multiply_then_reduce",
    )


def _inverse_metric_inner_reduction_path_error(
    settings: Mapping[str, Any],
) -> str | None:
    return _inner_reduction_path_error(
        settings,
        "inverse_metric_inner.reduction_path",
        "inverse_metric.solve_path",
        "solve_then_reduce",
    )


def _inner_reduction_path_error(
    settings: Mapping[str, Any],
    value_key: str,
    primary_key: str,
    primary_value: str,
) -> str | None:
    value = settings[value_key]

    if value == primary_value:
        return _path_dependency_error(
            settings,
            required_key=primary_key,
            required_message=f"{primary_value} requires {primary_key}",
            forbidden_keys={
                "sqrt_metric.factor_path": (
                    f"{primary_value} does not use sqrt_metric.factor_path"
                ),
            },
        )

    if value == "factored_gram":
        return _path_dependency_error(
            settings,
            forbidden_keys={
                primary_key: f"factored_gram does not use {primary_key}",
                "sqrt_metric.factor_path": (
                    "factored_gram does not use sqrt_metric.factor_path"
                ),
            },
        )

    if value == "sqrt_apply_reduce":
        return _path_dependency_error(
            settings,
            required_key="sqrt_metric.factor_path",
            required_message="sqrt_apply_reduce requires sqrt_metric.factor_path",
            forbidden_keys={
                primary_key: f"sqrt_apply_reduce does not use {primary_key}",
            },
        )

    return None


def _path_dependency_error(
    settings: Mapping[str, Any],
    *,
    required_key: str | None = None,
    required_message: str | None = None,
    forbidden_keys: Mapping[str, str] | None = None,
) -> str | None:
    if required_key is not None and required_key not in settings:
        if required_message is None:
            return f"{required_key} is required"

        return required_message

    if forbidden_keys is None:
        return None

    for key, message in forbidden_keys.items():
        if key in settings:
            return message

    return None


def _metric_inner_multi_rhs_axis(axis_key: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[axis_key]

        if value == "single_column":
            return True, None

        if value != "block":
            return False, f"{axis_key} is unsupported: {value}"

        if candidate.settings.get("vectorization.mode") not in {
            "single_loop",
            "manual_batch",
            "vmap",
        }:
            message = (
                f"{axis_key}=block requires vectorization.mode=single_loop, "
                "manual_batch, or vmap"
            )

            return False, message

        return True, None

    return admit


def _loss_scaling_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _loss_scaling_rule_error(candidate.settings, {})

        if error is not None:
            return False, error

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


def _per_example_gradient_accumulation_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        settings = candidate.settings
        accumulation = settings["per_example_gradient.accumulation"]

        if accumulation == "blockwise_stacked":
            if "batch.per_example_block_size" not in settings:
                return (
                    False,
                    "blockwise_stacked requires batch.per_example_block_size",
                )

            error = _positive_int_error(settings, "batch.per_example_block_size")

            return (True, None) if error is None else (False, error)

        if "batch.per_example_block_size" in settings:
            return (
                False,
                "batch.per_example_block_size requires blockwise_stacked",
            )

        return True, None

    return admit


def _per_example_block_size_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        settings = candidate.settings

        if settings.get("per_example_gradient.accumulation") != "blockwise_stacked":
            return (
                False,
                "batch.per_example_block_size requires blockwise_stacked",
            )

        error = _positive_int_error(settings, "batch.per_example_block_size")

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
    for key in PER_EXAMPLE_SCORE_PATH_KEYS:
        if key in settings:
            return settings[key]

    return None


def _manual_per_example_batch_size_error(settings: Mapping[str, Any]) -> str | None:
    key = _manual_per_example_batch_size_key(settings)

    if key is None:
        return "schedule.per_example=manual_batch requires a Fisher-family path"

    if key not in settings:
        return f"{key} is required for schedule.per_example=manual_batch"

    value = settings[key]

    if not _positive_int_value(value):
        return f"{key} must be a positive integer"

    return None


def _manual_per_example_batch_size_key(settings: Mapping[str, Any]) -> str | None:
    for path_keys, batch_key in MANUAL_PER_EXAMPLE_BATCH_SIZE_KEYS:
        if any(path_key in settings for path_key in path_keys):
            return batch_key

    return None


def _uses_ggn_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    if (
        settings.get("ggn.loss_hessian_kernel") == "dense_global"
        and "ggn.jvp_path" not in settings
    ):
        return True

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


def _uses_setting_path(
    settings: Mapping[str, Any],
    rows: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    return any(settings.get(key) in values for key, values in rows)


def _uses_composition_vector_execution(settings: Mapping[str, Any]) -> bool:
    return settings.get("composition.execution") in COMPOSITION_VECTOR_EXECUTIONS


def _uses_vector_loop_path(settings: Mapping[str, Any]) -> bool:
    return (
        _uses_setting_path(settings, VECTOR_LOOP_PATH_SETTINGS)
        or _uses_ggn_vector_loop_path(settings)
        or _uses_composition_vector_execution(settings)
    )


def _uses_vector_vmap_path(settings: Mapping[str, Any]) -> bool:
    return (
        _uses_setting_path(settings, VECTOR_VMAP_PATH_SETTINGS)
        or _uses_ggn_vector_vmap_path(settings)
        or _uses_composition_vector_execution(settings)
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


def _vectorization_randomness_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        settings = candidate.settings

        if settings.get("vectorization.mode") == "vmap":
            error = _vmap_randomness_error(settings)

            return (True, None) if error is None else (False, error)

        if _uses_torch_func_admission_path(settings):
            return True, None

        return (
            False,
            "vectorization.randomness requires vmap mode or torch.func lowering",
        )

    return admit


def _uses_torch_func_admission_path(settings: Mapping[str, Any]) -> bool:
    return any(
        settings.get(key) == value for key, value in TORCH_FUNC_ADMISSION_PATH_SETTINGS
    )


def _vmap_chunk_size_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.vmap_chunk_size"

    if key not in settings:
        return "vectorization.mode=vmap requires vectorization.vmap_chunk_size"

    chunk_size = settings[key]

    if not _positive_int_value(chunk_size):
        return "vectorization.vmap_chunk_size must be a positive integer"

    return None


def _manual_batch_size_error(settings: Mapping[str, Any]) -> str | None:
    key = "vectorization.batch_size"

    if key not in settings:
        return "vectorization.mode=manual_batch requires vectorization.batch_size"

    batch_size = settings[key]

    if not _positive_int_value(batch_size):
        return "vectorization.batch_size must be a positive integer"

    return None


def _per_token_schedule_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        schedule = candidate.settings["schedule.per_token"]

        if schedule != "packed":
            return True, None

        if candidate.settings.get("input.batch_layout") in {
            "packed_with_inverse_permutation",
            "variable_length",
        }:
            return True, None

        return False, (
            "schedule.per_token=packed requires input.batch_layout "
            "packed_with_inverse_permutation or variable_length"
        )

    return admit


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
    error = None

    if (isinstance(in_dims, int) and not isinstance(in_dims, bool)) or in_dims is None:
        error = None
    elif isinstance(in_dims, Mapping):
        error = _vmap_batch_in_dims_mapping_error(in_dims)
    elif isinstance(in_dims, tuple):
        error = _vmap_batch_in_dims_tuple_error(in_dims)
    else:
        error = (
            "vectorization.in_dims values must be integers, None, mappings, or tuples"
        )

    return error


def _vmap_batch_in_dims_mapping_error(in_dims: Mapping[Any, Any]) -> str | None:
    if not in_dims:
        return "vectorization.in_dims mapping must be nonempty"

    for key, value in in_dims.items():
        if not isinstance(key, str) or not key:
            return "vectorization.in_dims keys must be nonempty strings"

        error = _vmap_batch_in_dims_value_error(value)

        if error is not None:
            return error

    return None


def _vmap_batch_in_dims_tuple_error(in_dims: tuple[Any, ...]) -> str | None:
    if not in_dims:
        return "vectorization.in_dims tuple must be nonempty"

    for value in in_dims:
        error = _vmap_batch_in_dims_value_error(value)

        if error is not None:
            return error

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


StandardAxisRuleFactory = Callable[[tuple[str, ...]], AdmissionRule]


@dataclasses.dataclass(frozen=True, slots=True)
class _StandardAxisRow:
    """Input row for a standard core axis descriptor."""

    name: str
    settings_keys: tuple[str, ...]
    allowed_values: tuple[Any, ...]
    optional_settings_keys: tuple[str, ...] = ()
    rule_factory: StandardAxisRuleFactory | None = None


def _fixed_axis_rule(
    rule_factory: Callable[[], AdmissionRule],
) -> StandardAxisRuleFactory:
    def build(_: tuple[str, ...]) -> AdmissionRule:
        return rule_factory()

    return build


def _single_setting_axis_rule(
    rule_factory: Callable[[str], AdmissionRule],
) -> StandardAxisRuleFactory:
    def build(settings_keys: tuple[str, ...]) -> AdmissionRule:
        return rule_factory(settings_keys[0])

    return build


def _all_settings_axis_rule(
    rule_factory: Callable[..., AdmissionRule],
) -> StandardAxisRuleFactory:
    def build(settings_keys: tuple[str, ...]) -> AdmissionRule:
        return rule_factory(*settings_keys)

    return build


def _single_axis_row(
    name: str,
    allowed_values: tuple[Any, ...],
    rule_factory: StandardAxisRuleFactory | None = None,
    *,
    optional_settings_keys: tuple[str, ...] = (),
) -> _StandardAxisRow:
    return _StandardAxisRow(
        name,
        (name,),
        allowed_values,
        optional_settings_keys,
        rule_factory,
    )


def _single_axis_rows(
    names: tuple[str, ...],
    allowed_values: tuple[Any, ...],
    rule_factory: StandardAxisRuleFactory | None = None,
) -> tuple[_StandardAxisRow, ...]:
    return tuple(_single_axis_row(name, allowed_values, rule_factory) for name in names)


def _multi_axis_row(
    name: str,
    settings_keys: tuple[str, ...],
    allowed_values: tuple[Any, ...],
    rule_factory: StandardAxisRuleFactory | None = None,
    *,
    optional_settings_keys: tuple[str, ...] = (),
) -> _StandardAxisRow:
    return _StandardAxisRow(
        name,
        settings_keys,
        allowed_values,
        optional_settings_keys,
        rule_factory,
    )


def _standard_axis_from_row(row: _StandardAxisRow) -> AxisDescriptor:
    admission_rule = None

    if row.rule_factory is not None:
        admission_rule = row.rule_factory(row.settings_keys)

    return AxisDescriptor(
        row.name,
        row.settings_keys,
        row.allowed_values,
        optional_settings_keys=row.optional_settings_keys,
        admission_rule=admission_rule,
    )


LAYOUT_TREE_VALUES = (*DIRECT_LAYOUT_VALUES, *DISTRIBUTED_LAYOUT_VALUES)
SCORE_GRAD_PATH_VALUES = (
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "vmap_grad",
    "backward_materialized_grad",
)
HVP_PATH_VALUES = (
    "reverse_over_reverse",
    "jvp_grad",
    "autograd_functional_hvp",
    "autograd_functional_vhp",
    "forward_ad_dual",
    "linearize_grad",
)
RETAIN_OR_RECOMPUTE_VALUES = ("retain", "recompute")
FALSE_TRUE_VALUES = ("false", "true")
LAYER_BLOCK_VALUES = ("layer_blocks", "module_blocks", "custom_blocks")
SINGLE_OR_BLOCK_VALUES = ("single_column", "block")
METRIC_INNER_REDUCTION_VALUES = (
    "multiply_then_reduce",
    "factored_gram",
    "sqrt_apply_reduce",
)
INVERSE_METRIC_INNER_REDUCTION_VALUES = (
    "solve_then_reduce",
    "factored_gram",
    "sqrt_apply_reduce",
)
COMPILE_SETTING_RULE = _fixed_axis_rule(_compile_setting_axis)
POSITIVE_INT_RULE = _single_setting_axis_rule(_positive_int_axis)
STORAGE_COMPUTE_DTYPE_RULE = _single_setting_axis_rule(_storage_compute_dtype_axis)
LAYOUT_TREE_RULE = _single_setting_axis_rule(_layout_tree_axis)
MEMORY_OUTPUT_RULE = _single_setting_axis_rule(_memory_output_axis)
METRIC_INNER_MULTI_RHS_RULE = _single_setting_axis_rule(_metric_inner_multi_rhs_axis)
PATH_ADMISSION_RULE = _single_setting_axis_rule(_path_admission_axis)

STANDARD_AXIS_ROWS = (
    *_single_axis_rows(
        ("dtype.parameter_storage", "dtype.model_compute"),
        SPEC_STORAGE_COMPUTE_DTYPE_VALUES,
        STORAGE_COMPUTE_DTYPE_RULE,
    ),
    *_single_axis_rows(
        (
            "dtype.autodiff_compute",
            "dtype.accumulation",
            "dtype.vector",
            "dtype.intermediate",
            "dtype.metric_factor",
            "dtype.output",
        ),
        SPEC_DTYPE_VALUES,
    ),
    *_single_axis_rows(
        ("layout.params", "layout.vector", "layout.output"),
        LAYOUT_TREE_VALUES,
        LAYOUT_TREE_RULE,
    ),
    _single_axis_row("layout.contiguity", ("contiguous", "preserve_existing_strides")),
    _single_axis_row("layout.flatten_order", ("canonical_parameter_order",)),
    _single_axis_row("layout.vector_ops", ("python_loop", "foreach")),
    _single_axis_row("layout.aliasing", ("preserve_tied_weight_aliases",)),
    _single_axis_row(
        "layout.parametrizations",
        ("preserve_active_parametrizations",),
    ),
    _single_axis_row("call.path", ("functional_call", "stateful_module")),
    _single_axis_row("call.params", ("explicit_params", "module_params")),
    _single_axis_row("call.buffers", ("explicit_buffers", "module_buffers")),
    _single_axis_row("call.tied_weights", ("preserve_alias_groups",)),
    _single_axis_row("call.parametrizations", ("preserve_parametrizations",)),
    _single_axis_row("call.buffer_mutation", ("forbidden", "declared_and_restored")),
    _single_axis_row("call.grad_mode", ("grad_enabled",)),
    _single_axis_row(
        "call.return_type",
        ("raw_tensor_tree", "model_output_object_with_declared_fields"),
    ),
    _single_axis_row("hvp.path", HVP_PATH_VALUES, PATH_ADMISSION_RULE),
    _single_axis_row(
        "hvp.graph_schedule",
        ("retain_graph_across_vectors", "rebuild_graph_per_vector"),
    ),
    _single_axis_row("hvp.primal_reuse", ("reuse_primal", "recompute_primal")),
    _single_axis_row(
        "hvp.gradient_reuse",
        ("reuse_gradient_closure", "recompute_gradient"),
    ),
    _single_axis_row(
        "gradient.path",
        (
            "torch_autograd_grad",
            "torch_func_grad",
            "torch_func_grad_and_value",
            "backward_materialized_grad",
        ),
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row(
        "gradient.value_reuse",
        ("gradient_only", "gradient_and_primal_value"),
        _fixed_axis_rule(_gradient_value_reuse_axis),
    ),
    _single_axis_row("gradient.graph_schedule", ("build_once", "rebuild_per_call")),
    _single_axis_row("jvp.path", JVP_VECTOR_LOOP_PATHS, PATH_ADMISSION_RULE),
    _single_axis_row(
        "jvp.linearize_reuse",
        ("none", "reuse_at_same_primal"),
        _fixed_axis_rule(_jvp_linearize_reuse_axis),
    ),
    _single_axis_row("vjp.path", VJP_VECTOR_LOOP_PATHS, PATH_ADMISSION_RULE),
    _single_axis_row(
        "vjp.closure_reuse",
        ("none", "reuse_vjp_closure_at_same_primal"),
        _fixed_axis_rule(_vjp_closure_reuse_axis),
    ),
    _single_axis_row(
        "ggn.jvp_path",
        JVP_VECTOR_LOOP_PATHS,
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row(
        "ggn.loss_hessian_path",
        ("closed_form_softmax_ce_kl", "autodiff_loss_hvp"),
    ),
    _single_axis_row(
        "ggn.loss_hessian_kernel",
        ("dense_global", "streaming_global", "two_pass_chunked_global"),
    ),
    _single_axis_row(
        "ggn.vjp_path",
        ("torch_func_vjp", "autograd_grad_outputs"),
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row("ggn.jvp_reuse", ("reuse_jvp", "recompute_jvp")),
    _single_axis_row(
        "ggn.cotangent_reuse",
        ("reuse_output_cotangent", "recompute_output_cotangent"),
    ),
    _single_axis_row("fisher.accumulation", FISHER_VECTOR_ACCUMULATIONS),
    _single_axis_row(
        "fisher.expectation_path",
        ("explicit_full_expectation_score_rows",),
    ),
    _single_axis_row(
        "fisher.score_grad_path",
        SCORE_GRAD_PATH_VALUES,
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row("sampled_fisher.accumulation", FISHER_VECTOR_ACCUMULATIONS),
    _single_axis_row(
        "sampled_fisher.sample_source",
        ("fixed_sample_table", "fixed_seed_and_count"),
    ),
    _single_axis_row(
        "sampled_fisher.score_grad_path",
        SCORE_GRAD_PATH_VALUES,
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row(
        "sampled_fisher.exact_fisher_check",
        ("disabled", "enabled_with_sampling_bound"),
    ),
    _single_axis_row(
        "empirical_fisher.grad_path",
        SCORE_GRAD_PATH_VALUES,
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row(
        "empirical_fisher.accumulation",
        (
            "streaming_dot_accumulate",
            "materialize_per_example_gradients",
            "blockwise_gradient_matrix",
        ),
    ),
    _single_axis_row(
        "per_example_gradient.grad_path",
        SCORE_GRAD_PATH_VALUES,
        PATH_ADMISSION_RULE,
    ),
    _single_axis_row(
        "per_example_gradient.accumulation",
        ("stacked_leading_axis", "blockwise_stacked"),
        _fixed_axis_rule(_per_example_gradient_accumulation_axis),
    ),
    _multi_axis_row(
        "forward_ad_flags",
        FORWARD_AD_FIELDS,
        (),
        _all_settings_axis_rule(_bool_axis),
    ),
    _multi_axis_row(
        "torch_func_admission",
        TORCH_FUNC_AXIS_FIELDS,
        (),
        _fixed_axis_rule(_torch_func_axis),
    ),
    _single_axis_row(
        "vectorization.mode",
        ("single_loop", "manual_batch", "vmap"),
        _fixed_axis_rule(_vectorization_mode_axis),
    ),
    _single_axis_row(
        "vectorization.randomness",
        ("error", "same", "different"),
        _fixed_axis_rule(_vectorization_randomness_axis),
    ),
    _single_axis_row("vectorization.batch_size", (), POSITIVE_INT_RULE),
    _single_axis_row(
        "vectorization.vmap_chunk_size",
        (),
        _fixed_axis_rule(_vmap_chunk_size_axis),
    ),
    _single_axis_row(
        "vectorization.in_dims",
        (),
        _fixed_axis_rule(_vmap_batch_in_dims_axis),
    ),
    _single_axis_row(
        "batch.data_microbatch_size",
        (),
        _fixed_axis_rule(_gradient_accumulation_axis),
    ),
    *_single_axis_rows(
        (
            "batch.hvp_row_batch_size",
            "batch.ggn_batch_size",
            "batch.fisher_sample_batch_size",
            "batch.empirical_example_batch_size",
        ),
        (),
        POSITIVE_INT_RULE,
    ),
    _single_axis_row(
        "batch.per_example_block_size",
        (),
        _fixed_axis_rule(_per_example_block_size_axis),
    ),
    _single_axis_row(
        "schedule.per_example",
        ("loop", "vmap", "manual_batch"),
        _fixed_axis_rule(_per_example_schedule_axis),
    ),
    _single_axis_row(
        "schedule.gradient_accumulation",
        ("single_step", "microbatch_accumulate"),
        _fixed_axis_rule(_gradient_accumulation_axis),
    ),
    _single_axis_row(
        "schedule.per_token",
        ("loop", "packed"),
        _fixed_axis_rule(_per_token_schedule_axis),
    ),
    *_single_axis_rows(
        ("chunk.token_block_size", "chunk.sequence_position_block_size"),
        (),
        POSITIVE_INT_RULE,
    ),
    _single_axis_row(
        "input.batch_layout",
        ("dense_padded", "packed_with_inverse_permutation", "variable_length"),
    ),
    _single_axis_row("input.length_grouping", ("none", "exact_length_bucket")),
    _single_axis_row(
        "input.host_to_device",
        ("outside_measured_call", "inside_measured_call"),
    ),
    _single_axis_row("input.residency", ("cpu_staged", "cpu_pinned", "gpu")),
    _single_axis_row(
        "teacher_outputs",
        (
            "precomputed_cpu",
            "precomputed_cpu_pinned",
            "precomputed_gpu",
            "recomputed_with_equality_check",
        ),
    ),
    _single_axis_row(
        "memory.vector_residency",
        ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu"),
    ),
    _single_axis_row(
        "memory.intermediate_residency",
        ("gpu", "cpu_pinned", "cpu_staged"),
        _fixed_axis_rule(_memory_intermediate_residency_axis),
    ),
    _single_axis_row(
        "memory.factor_residency",
        ("gpu", "cpu_pinned", "cpu_staged", "mmap_cpu"),
    ),
    _single_axis_row(
        "memory.output_buffers",
        ("fresh_allocation", "preallocated"),
    ),
    *_single_axis_rows(
        (
            "memory.primal_outputs",
            "memory.jvp_outputs",
            "memory.output_cotangents",
        ),
        RETAIN_OR_RECOMPUTE_VALUES,
        MEMORY_OUTPUT_RULE,
    ),
    *_single_axis_rows(
        (
            "chunk.class_block_size_with_exact_global_normalization",
            "chunk.output_cotangent_block_size",
            "chunk.parameter_block_size",
            "chunk.layer_block_size",
            "chunk.lm_head_weight_chunk_bytes",
        ),
        (),
        POSITIVE_INT_RULE,
    ),
    _single_axis_row("compile.enabled", FALSE_TRUE_VALUES, COMPILE_SETTING_RULE),
    _single_axis_row(
        "compile.boundary",
        COMPILE_BOUNDARY_VALUES,
        _fixed_axis_rule(_compile_boundary_axis),
    ),
    _single_axis_row("compile.backend", (), _fixed_axis_rule(_compile_backend_axis)),
    _single_axis_row(
        "compile.mode",
        (None, "default", "max-autotune"),
        COMPILE_SETTING_RULE,
    ),
    _single_axis_row("compile.fullgraph", FALSE_TRUE_VALUES, COMPILE_SETTING_RULE),
    _single_axis_row("compile.dynamic", (None, "false", "true"), COMPILE_SETTING_RULE),
    *_single_axis_rows(
        (
            "compile.compiled_autograd",
            "compile.options.epilogue_fusion",
            "compile.options.shape_padding",
            "compile.cuda_graphs",
        ),
        FALSE_TRUE_VALUES,
        COMPILE_SETTING_RULE,
    ),
    _single_axis_row(
        "compile.cache_state",
        ("cold_compile", "warm_cache"),
        COMPILE_SETTING_RULE,
    ),
    _single_axis_row(
        "activation.recompute",
        (
            "none",
            "checkpoint_non_reentrant_by_layer",
            "checkpoint_selective",
            "manual_recompute",
        ),
    ),
    _single_axis_row(
        "activation.offload",
        ("none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"),
        _fixed_axis_rule(_activation_offload_axis),
        optional_settings_keys=("activation.pack_hook", "activation.unpack_hook"),
    ),
    _single_axis_row(
        "checkpoint.use_reentrant",
        ("false",),
        optional_settings_keys=(
            "checkpoint.moves_to_new_device",
            "checkpoint.uses_global_state",
        ),
    ),
    *_single_axis_rows(
        ("checkpoint.early_stop", "checkpoint.preserve_rng_state"),
        FALSE_TRUE_VALUES,
    ),
    _single_axis_row("checkpoint.determinism_check", ("default", "none")),
    _single_axis_row(
        "checkpoint.context_fn",
        ("none", "declared_context_pair"),
        _fixed_axis_rule(_checkpoint_context_axis),
        optional_settings_keys=("checkpoint.context_fn_callable",),
    ),
    _single_axis_row("numeric.float32_matmul_precision", MATMUL_PRECISION_VALUES),
    _single_axis_row("autocast", ("off", "cuda_fp16", "cuda_bf16")),
    _single_axis_row(
        "metric.multiply_path",
        (
            "dense_matmul",
            "factorized_multiply",
            "blockwise_multiply",
            "streaming_multiply",
        ),
    ),
    _single_axis_row("metric.accumulation", ("streaming", "materialized_blocks")),
    _single_axis_row("metric.block_schedule", LAYER_BLOCK_VALUES),
    _single_axis_row(
        "metric_inner.reduction_path",
        METRIC_INNER_REDUCTION_VALUES,
        _fixed_axis_rule(_metric_inner_reduction_path_axis),
    ),
    _single_axis_row(
        "metric_inner.multi_rhs",
        SINGLE_OR_BLOCK_VALUES,
        METRIC_INNER_MULTI_RHS_RULE,
    ),
    _single_axis_row(
        "sqrt_metric.factor_path",
        (
            "closed_form_factor_square_root",
            "cholesky_factor",
            "eigenbasis_factor",
            "matrix_free_lanczos",
        ),
        _fixed_axis_rule(_sqrt_metric_factor_path_axis),
    ),
    _single_axis_row(
        "sqrt_metric.lanczos_iterations",
        (),
        _fixed_axis_rule(_sqrt_metric_lanczos_iterations_axis),
    ),
    _single_axis_row("inverse_metric.solve_path", INVERSE_METRIC_VECTOR_LOOP_PATHS),
    _single_axis_row(
        "inverse_metric.preconditioner",
        ("none", "diagonal", "block_diagonal", "factorized_metric", "matrix_free"),
        _fixed_axis_rule(_inverse_metric_preconditioner_axis),
        optional_settings_keys=("inverse_metric.preconditioner_product",),
    ),
    _single_axis_row("inverse_metric.iteration_budget", (), POSITIVE_INT_RULE),
    _single_axis_row(
        "inverse_metric.factor_reuse",
        ("refactor_each_rhs", "reuse_factor_across_rhs"),
    ),
    _single_axis_row("inverse_metric.block_schedule", LAYER_BLOCK_VALUES),
    _single_axis_row(
        "inverse_metric.multi_rhs",
        SINGLE_OR_BLOCK_VALUES,
        _fixed_axis_rule(_inverse_metric_multi_rhs_axis),
    ),
    _single_axis_row(
        "inverse_metric_inner.reduction_path",
        INVERSE_METRIC_INNER_REDUCTION_VALUES,
        _fixed_axis_rule(_inverse_metric_inner_reduction_path_axis),
    ),
    _single_axis_row(
        "inverse_metric_inner.multi_rhs",
        SINGLE_OR_BLOCK_VALUES,
        METRIC_INNER_MULTI_RHS_RULE,
    ),
    _single_axis_row(
        "composition.execution",
        (
            "materialize_each_child",
            "stream_child_outputs",
            "fuse_adjacent_children",
            "compile_whole_composition",
        ),
        _fixed_axis_rule(_composition_execution_axis),
    ),
    _single_axis_row(
        "composition.child_evaluation",
        ("selected_child_rows", "inline_child_lowering"),
    ),
    _single_axis_row(
        "composition.validation",
        ("validate_each_child", "validate_composed_output"),
    ),
    *_single_axis_rows(
        (
            "numeric.bf16_reduced_precision_reduction",
            "numeric.fp16_reduced_precision_reduction",
        ),
        FALSE_TRUE_VALUES,
    ),
    _single_axis_row(
        "fusion.norm",
        ("model_default", "fused_rmsnorm", "fused_layernorm"),
    ),
    _single_axis_row("fusion.mlp", ("model_default", "fused_mlp")),
    _single_axis_row("fusion.rope", ("model_default", "fused_rope")),
    _single_axis_row("fusion.logits", ("model_default", "fused_logits_projection")),
    _single_axis_row("fusion.loss", ("model_default", "fused_ce", "fused_kl")),
    _single_axis_row("numeric.deterministic_algorithms", FALSE_TRUE_VALUES),
    _single_axis_row(
        "numeric.loss_scaling",
        ("none", "static_scale_with_exact_unscale"),
        _fixed_axis_rule(_loss_scaling_axis),
        optional_settings_keys=("numeric.loss_scale", "numeric.loss_unscale_degree"),
    ),
)


def standard_axis_descriptors() -> tuple[AxisDescriptor, ...]:
    """Return standard core axis descriptors."""
    return tuple(_standard_axis_from_row(row) for row in STANDARD_AXIS_ROWS)


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
    generator_version: str | None = None,
) -> tuple[Candidate, ...]:
    """Create candidates from an ordered grid of axis values.

    Returns:
        Candidate grid in deterministic order.
    """
    items = tuple(axes.items())
    candidates = []
    candidate_generator_version = (
        PACKAGE_VERSION if generator_version is None else generator_version
    )

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
                    generator_version=candidate_generator_version,
                )
            )

            return

        axis_name, values = items[index]

        if not values:
            message = f"candidate axis has no values: {axis_name}"
            raise AdmissionError(message)

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
