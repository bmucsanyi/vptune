import ast
import contextlib
import dataclasses
import math
import re
import tomllib
import types
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, override

import autobatch
import pytest
import torch
from torch.nn.utils import parametrize
from vptune_test_helpers import (
    assert_tree_close,
    thresholds_for_measurements,
)

import vptune as vp
import vptune.adapters as vpa
import vptune.axes.candidates as candidates_module
import vptune.engine.attention as attention_module
import vptune.engine.runtime as runtime_module
import vptune.ext as vpx
import vptune.tuning.measure as measure_module
import vptune.tuning.run as run_module
from vptune.core import operators as ops
from vptune.core.data import PACKAGE_VERSION, FullSizeRecord, Measurement
from vptune.core.identities import (
    canonical_json,
    cuda_driver_version,
    module_identity,
    stable_hash,
    to_json_value,
)
from vptune.core.tensor_tree import (
    tree_add_foreach,
    tree_dot_foreach,
    tree_elementwise_div_foreach,
    tree_elementwise_mul_foreach,
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_mul_foreach,
    tree_signature,
)
from vptune.engine.checks import (
    numeric_error_bound_measurements,
    uses_reduction_degrading_setting,
    validate_numeric_error_bound,
    validate_thresholds,
)
from vptune.errors import ReferenceFailedError
from vptune.tuning import autobatch_bridge
from vptune.tuning.io import read_record, write_record, write_record_exclusive
from vptune.tuning.measure import (
    CPUMemoryBackend,
    measure_once,
    measure_operation,
    run_candidate,
)
from vptune.tuning.run import tune as tune_problem
from vptune.tuning.schemas import record_current
from vptune.tuning.select import memory_stable, select_cohort, select_family

MANIFEST_AXIS_BULLET = re.compile(r"^- `([^`]+)`(?:: (.*))?$")
MANIFEST_AXIS_KEY_ALLOWLIST = {"autocast", "teacher_outputs"}
SPEC_MANIFEST_START = "The package manifest must include these operator-owned axes:"
SPEC_MANIFEST_END = "These manifest rules reject contradictory rows:"
SPEC_ACCEPTANCE_START = "## Acceptance Tests"
SPEC_ACCEPTANCE_END = "## Package Layout"
SPEC_SYMBOLIC_AXIS_DOMAINS = {
    "compile.backend": ("inductor", "registered_backend"),
    "vectorization.in_dims": candidates_module.DECLARED_DOMAIN,
    "attention.custom_kernel_id": candidates_module.REGISTERED_DOMAIN,
    "attention.mask_formatter_id": candidates_module.REGISTERED_DOMAIN,
    "distributed.mesh_dim_names": candidates_module.DECLARED_DOMAIN,
    "fsdp.reshard_after_forward": (candidates_module.FSDP_RESHARD_AFTER_FORWARD_DOMAIN),
    "fsdp.ignored_params": candidates_module.DECLARED_DOMAIN,
    "fsdp.dp_mesh_dims": candidates_module.DECLARED_DOMAIN,
    "tp.plan": candidates_module.REGISTERED_DOMAIN,
    "tp.prepare_module_input": candidates_module.DECLARED_DOMAIN,
    "tp.prepare_module_output": candidates_module.DECLARED_DOMAIN,
    "sequence_parallel.norm_modules": candidates_module.DECLARED_DOMAIN,
    "context_parallel.sequence_dim": candidates_module.DECLARED_DOMAIN,
}
MANIFEST_SYMBOLIC_VALUE_TEST_COVERAGE = {
    ("layout.vector_ops", "python_loop"): (
        "test_standard_runtime_executes_foreach_vector_ops_for_diagonal_metric",
    ),
    ("hvp.gradient_reuse", "recompute_gradient"): (
        "test_hvp_reverse_reuse_vector_policies",
    ),
    ("gradient.value_reuse", "gradient_only"): (
        "test_gradient_reference_check_records_directional_agreement",
    ),
    ("vectorization.batch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_vectorization_manual_batch_settings_are_validated",
    ),
    ("vectorization.vmap_chunk_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_hvp_vectorization_runs_batched_vectors",
    ),
    ("batch.data_microbatch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_standard_runtime_accepts_direct_input_schedule_settings",
    ),
    ("batch.hvp_row_batch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_hvp_row_batch_size_executes_reverse_rows",
    ),
    ("batch.ggn_batch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_ggnvp_chunks_output_cotangent_vjp",
    ),
    ("batch.fisher_sample_batch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_fisher_family_vectorization_runs_batched_vectors",
    ),
    ("batch.empirical_example_batch_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_empirical_fisher_vmap_path_matches_loop_path",
    ),
    ("batch.per_example_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_per_example_gradient_stacked_and_blockwise_execute_declared_rows",
    ),
    ("chunk.token_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_ggnvp_closed_form_ce_kl_token_blocks_match_dense_loss_hessian",
    ),
    ("chunk.sequence_position_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_attention_operation_uses_candidate_sequence_block_size",
    ),
    (
        "chunk.class_block_size_with_exact_global_normalization",
        candidates_module.INTEGER_DOMAIN[0],
    ): ("test_ggnvp_closed_form_ce_kl_token_blocks_match_dense_loss_hessian",),
    ("chunk.output_cotangent_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_ggnvp_chunks_output_cotangent_vjp",
    ),
    ("chunk.parameter_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_parameter_block_size_rejects_non_dense_parameter_matrix_path",
    ),
    ("chunk.layer_block_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_layer_block_size_requires_declared_layer_groups",
    ),
    ("chunk.lm_head_weight_chunk_bytes", candidates_module.INTEGER_DOMAIN[0]): (
        "test_standard_runtime_executes_lm_head_chunker_binding",
    ),
    ("compile.boundary", "metric_inner_reduce"): (
        "test_standard_runtime_compiles_metric_tree_boundary_only",
    ),
    ("compile.boundary", "metric_sqrt_multiply"): (
        "test_standard_runtime_compiles_metric_tree_boundary_only",
    ),
    ("compile.boundary", "inverse_metric_inner_reduce"): (
        "test_standard_runtime_compiles_metric_tree_boundary_only",
    ),
    ("sqrt_metric.lanczos_iterations", candidates_module.INTEGER_DOMAIN[0]): (
        "test_matrix_free_lanczos_requires_declared_iterations",
    ),
    ("inverse_metric.iteration_budget", candidates_module.INTEGER_DOMAIN[0]): (
        "test_inverse_metric_cg_stops_at_declared_tol",
    ),
    ("attention.frontend", "paged|eager"): (
        "test_transformers_attention_axis_uses_core_and_adapter_admission_fields",
    ),
    ("attention.frontend", "paged|sdpa"): (
        "test_transformers_attention_axis_uses_core_and_adapter_admission_fields",
    ),
    ("distributed.mesh_shape", candidates_module.INTEGER_TUPLE_DOMAIN[0]): (
        "test_distributed_identity_records_mesh_and_communication",
    ),
    (
        "fsdp.reshard_after_forward",
        candidates_module.FSDP_RESHARD_AFTER_FORWARD_DOMAIN[0],
    ): ("test_distributed_adapter_registry_admits_direct_fsdp_reshard_group_sizes",),
    ("comm.collective_bucket_size", candidates_module.INTEGER_DOMAIN[0]): (
        "test_distributed_strategy_applier_lowers_fsdp2_row_settings",
    ),
}
MANIFEST_CHECK_TEST_COVERAGE = {
    "attention_backend_equality": (
        "test_attention_operation_factory_and_reference_check_execute_core_row",
    ),
    "autograd_functional_anchor": (
        "test_hvp_reference_check_accepts_functional_hvp_path",
    ),
    "checkpoint_recompute_equality": (
        "test_standard_runtime_executes_selective_checkpoint_with_context_pair",
    ),
    "child_anchor_checks": ("test_composition_reference_check_runs_child_anchor",),
    "dense_composed_output_check": (
        "test_composition_validate_composed_output_skips_child_anchor",
    ),
    "dense_empirical_fisher_anchor": (
        "test_per_example_gradient_reference_check_and_empirical_outer_product",
    ),
    "dense_factor_check": ("test_sqrt_metric_cholesky_paths_match_reference",),
    "dense_fisher_anchor": (
        "test_fisher_references_use_declared_per_example_objectives",
    ),
    "dense_inverse_gram_reference": (
        "test_inverse_metric_inner_solve_path_matches_reference",
    ),
    "dense_inverse_reference": (
        "test_inverse_metric_reference_check_records_inverse_residual",
    ),
    "dense_jacobian_ggn_anchor": (
        "test_ggnvp_reference_check_records_dense_anchor_errors",
    ),
    "dense_metric_gram_reference": ("test_metric_inner_dense_paths_match_reference",),
    "dense_metric_reference": (
        "test_metric_reference_check_rejects_nonsymmetric_metric",
    ),
    "dense_sampled_fisher_anchor": (
        "test_sampled_fisher_vp_dense_loop_and_anchor_use_parameter_order",
    ),
    "dependency_identity_equality": ("test_tune_run_uses_family_dag_order",),
    "direct_autograd_anchor": (
        "test_gradient_reference_check_records_directional_agreement",
    ),
    "distributed_logical_output_agreement": (
        "test_distributed_reference_check_uses_single_device_anchor",
    ),
    "dtype_reference_agreement": (
        "test_standard_reference_check_low_precision_anchor",
    ),
    "empirical_fisher_outer_product_check": (
        "test_per_example_gradient_reference_check_and_empirical_outer_product",
    ),
    "explicit_sampled_score_outer_product_anchor": (
        "test_sampled_fisher_vp_dense_loop_and_anchor_use_parameter_order",
    ),
    "explicit_score_outer_product_anchor": (
        "test_fisher_references_use_declared_per_example_objectives",
    ),
    "finite_difference_directional": (
        "test_jvp_reference_check_records_finite_difference_agreement",
    ),
    "finite_difference_gradient_directional": (
        "test_gradient_reference_check_records_directional_agreement",
    ),
    "fixed_sample_source_check": ("test_typed_sampled_fisher_fixed_seed_repeats",),
    "full_size_agreement": ("test_tune_measures_every_probe_input",),
    "fused_kernel_reference_agreement": (
        "test_composition_fuse_adjacent_children_reference_validates_fused_output",
    ),
    "hvp_symmetry": ("test_hvp_reference_check_records_symmetry_error",),
    "input_representation_equality": (
        "test_standard_runtime_executes_packed_input_layout_binding",
    ),
    "inverse_inner_residual_check": (
        "test_inverse_metric_inner_cg_uses_declared_tol_before_reduction",
    ),
    "inverse_residual_check": (
        "test_inverse_metric_reference_check_records_inverse_residual",
    ),
    "jvp_anchor": ("test_jvp_reference_check_records_finite_difference_agreement",),
    "jvp_hessian_vjp_cross_check": (
        "test_ggnvp_reference_check_uses_jvp_hessian_vjp_anchor",
    ),
    "jvp_vjp_dot_identity": ("test_vjp_reference_check_records_dot_identity",),
    "layout_roundtrip_reference": (
        "test_standard_runtime_executes_layout_output_flat_contiguous",
    ),
    "loss_hessian_psd": ("test_ggnvp_reference_check_rejects_indefinite_loss_hessian",),
    "loss_hessian_symmetry": (
        "test_ggnvp_reference_check_rejects_nonsymmetric_loss_hessian",
    ),
    "matrix_free_covariance_check": (
        "test_public_matrix_free_metric_square_roots_tune_selected_curvature_product",
    ),
    "metric_inner_diagonal_nonnegative_check": (
        "test_metric_inner_norm_requires_sqrt_apply_reduce",
    ),
    "metric_psd_check": ("test_metric_reference_check_rejects_indefinite_metric",),
    "metric_symmetry_check": (
        "test_metric_reference_check_rejects_nonsymmetric_metric",
    ),
    "numeric_error_bound_check": (
        "test_standard_reference_check_applies_numeric_error_bound_fields",
    ),
    "per_example_gradient_loop_anchor": (
        "test_per_example_gradient_reference_check_and_empirical_outer_product",
    ),
    "recompute_or_offload_equality": (
        "test_activation_offload_preserves_higher_order_hvp",
    ),
    "reverse_over_reverse_anchor": ("test_gradient_jvp_vjp_hvp_anchors",),
    "segmentation_invariance": (
        "test_segmented_forward_ad_attention_matches_full_attention_tangent",
    ),
    "teacher_output_equality": (
        "test_standard_runtime_rejects_mismatched_recomputed_teacher_outputs",
    ),
    "vjp_anchor": ("test_vjp_reference_check_records_dot_identity",),
}
ACCEPTANCE_TEST_COVERAGE = {
    "Axis manifest contains every key and value": (
        "test_axis_manifest_matches_spec_key_and_value_domains",
        "test_axis_manifest_keys_match_features_and_spec",
        "test_axis_manifest_carries_lowering_and_check_identity_fields",
    ),
    "Axis manifest rejects contradictory rows for packing": (
        "test_standard_axis_registry_validates_core_axes",
        "test_core_attention_axis_rejects_invalid_rows",
        "test_build_dtensor_placement_rejects_contradictory_fields",
        "test_manifest_rejects_declared_contradictory_rows",
        "test_memory_output_recompute_rejects_unmatched_settings",
        "test_checkpoint_operation_rejects_missing_activation_offload",
        "test_checkpoint_operation_rejects_custom_offload_without_hooks",
        "test_standard_runtime_rejects_input_schedule_rows_without_binding",
        "test_candidate_rows_reject_sampled_fisher_exact_check_without_bound",
        "test_sampled_fisher_vp_rejects_inconsistent_rows",
        "test_typed_categorical_fisher_routes_to_ggn",
    ),
    "Axis manifest rejects `compile.options.*=true`": (
        "test_standard_axis_registry_validates_core_axes",
        "test_standard_runtime_executes_dtype_and_backend_axes",
    ),
    "Axis manifest rejects metric and inverse-metric rows": (
        "test_candidate_rows_reject_metric_representation_path_mismatches",
        "test_non_dense_metric_paths_require_accumulation",
        "test_standard_runtime_executes_metric_factor_residency_axis",
    ),
    "`vp.problem(...)` and `vp.autotune(...)` reject composition": (
        "test_public_single_product_entrypoints_reject_composition",
    ),
    "Operator constructors validate the closed-set fields": (
        "test_typed_softmax_cross_entropy_rejects_invalid_fields",
        "test_typed_sample_source_validation",
        "test_typed_kfac_rejects_invalid_factor_declarations",
    ),
    "Every manifest value has one admission rule": (
        "test_axis_manifest_values_and_checks_have_test_coverage",
    ),
    "Gradient anchor matches direct autograd": ("test_gradient_jvp_vjp_hvp_anchors",),
    "JVP anchor matches finite difference": (
        "test_jvp_reference_check_records_finite_difference_agreement",
    ),
    "VJP anchor satisfies the dot-product identity": (
        "test_vjp_reference_check_records_dot_identity",
    ),
    "HVP anchor matches reverse-over-reverse": (
        "test_gradient_jvp_vjp_hvp_anchors",
        "test_gradient_reference_check_records_directional_agreement",
    ),
    "Standard runtime builder runs gradient, JVP": (
        "test_standard_operation_factory_runs_core_derivative_products",
    ),
    "Standard runtime builder runs dense GGNVP": (
        "test_standard_operation_factory_runs_dense_metric_and_fisher_families",
        "test_metric_inner_dense_paths_match_reference",
        "test_inverse_metric_inner_reference_check_records_inverse_residual",
    ),
    "Standard runtime applies declared `dtype.parameter_storage`": (
        "test_standard_runtime_executes_dtype_and_backend_axes",
        "test_standard_runtime_executes_split_model_and_autodiff_compute_dtypes",
    ),
    "Grad-materialization tests cover tensor-tree returns": (
        "test_standard_operation_factory_runs_core_derivative_products",
        "test_standard_runtime_executes_stateful_module_gradient",
    ),
    "Teacher-output tests cover CPU": (
        "test_standard_runtime_executes_precomputed_cpu_teacher_outputs",
        "test_standard_runtime_executes_precomputed_pinned_teacher_outputs",
        "test_standard_runtime_executes_precomputed_gpu_teacher_outputs",
        "test_standard_runtime_rejects_mismatched_recomputed_teacher_outputs",
    ),
    "Numeric loss-scaling tests cover degree-one unscale": (
        "test_numeric_loss_scaling_scales_gradient_source_and_unscales_output",
        "test_numeric_loss_scaling_unscales_degree_two_fisher_family_output",
        "test_numeric_loss_scaling_rejects_wrong_operator_degree",
    ),
    "Fusion tests cover every `fusion.*` axis value": (
        "test_standard_runtime_accepts_model_default_fusion_settings",
        "test_standard_runtime_executes_fused_row_with_registered_rewriter",
        "test_standard_runtime_executes_fused_rows_for_higher_order_families",
        "test_composition_fuse_adjacent_children_reference_validates_fused_output",
    ),
    "Activation-offload tests cover CPU saved-tensor hooks": (
        "test_standard_runtime_executes_cpu_saved_tensor_hooks",
        "test_standard_runtime_executes_custom_saved_tensor_hooks",
        "test_activation_offload_preserves_higher_order_hvp",
    ),
    "Layout tests cover `layout.vector_ops=foreach`": (
        "test_standard_runtime_executes_foreach_vector_ops_for_diagonal_metric",
        "test_standard_runtime_executes_layout_contiguity_axis",
        "test_standard_runtime_preserves_tied_parameter_aliases_during_dtype_cast",
        "test_distributed_operation_factory_delegates_dtensor_layout_to_strategy",
        "test_standard_runtime_executes_layout_output_flat_contiguous",
        "test_standard_runtime_executes_layout_params_flat_contiguous",
        "test_standard_runtime_executes_layout_vector_flat_contiguous",
        "test_standard_runtime_executes_layout_output_alias_groups",
    ),
    "Standard runtime rejects registered axes whose execution belongs": (
        "test_distributed_layout_values_require_distributed_adapter",
        "test_standard_compile_boundary_rejects_transformer_block_without_adapter",
    ),
    "`vhp` candidate path reports an HVP result": (
        "test_vhp_reference_check_requires_symmetry_and_directional_checks",
        "test_hvp_vhp_path_supports_parameter_tree_order",
    ),
    "GGNVP cross-checks dense": (
        "test_ggnvp_reference_check_uses_jvp_hessian_vjp_anchor",
        "test_ggnvp_reference_check_cross_checks_jvp_path_with_dense_anchor",
    ),
    "GGNVP enforces PSD on the output-space loss Hessian": (
        "test_ggnvp_reference_check_rejects_nonsymmetric_loss_hessian",
        "test_ggnvp_reference_check_rejects_indefinite_loss_hessian",
        "test_typed_declared_psd_matrix_free_rejects_indefinite_matvec",
    ),
    "FisherVP anchor computes exact score-gradient outer products": (
        "test_fisher_references_use_declared_per_example_objectives",
    ),
    "Exact categorical NLL Fisher is expressed by GGNVP": (
        "test_typed_categorical_fisher_routes_to_ggn",
        "test_ggnvp_reference_check_uses_jvp_hessian_vjp_anchor",
    ),
    "Sampled FisherVP uses declared fixed sample table": (
        "test_typed_sampled_fisher_fixed_seed_repeats",
        "test_sampled_fisher_vp_rejects_inconsistent_rows",
        "test_sampled_fisher_vp_dense_loop_and_anchor_use_parameter_order",
        "test_sampled_fisher_exact_comparison_runs_only_with_declared_bound",
    ),
    "EmpiricalFisherVP anchor computes per-example-gradient": (
        "test_per_example_gradient_reference_check_and_empirical_outer_product",
    ),
    "EmpiricalFisherVP standard runtime has both loop": (
        "test_empirical_fisher_vmap_path_matches_loop_path",
        "test_empirical_fisher_vmap_path_rejects_invalid_batch_shape",
        "test_vector_vmap_chunk_size_reaches_torch_func_vmap",
    ),
    "Standard dense metric materialization returns one object": (
        "test_standard_metric_materializer_returns_metric_object",
        "test_inverse_metric_materializer_calls_inverse_by_default",
    ),
    "Metric tests cover dense, diagonal, block-diagonal": (
        "test_typed_dense_metric_products_execute_against_reference",
        "test_typed_diagonal_metric_products_execute_against_reference",
        "test_typed_block_metric_products_execute_against_reference",
        "test_typed_kfac_metric_products_execute_against_reference",
        "test_typed_low_rank_metric_products_execute_against_reference",
        "test_typed_ggn_derived_metric_products_execute_against_reference",
    ),
    "Inverse-metric tests cover every solve path": (
        "test_inverse_metric_dense_direct_solve_paths_match_dense_solve",
        "test_inverse_metric_cg_dense_preconditioners_match_reference",
        "test_inverse_metric_direct_solve_rejects_iteration_budget",
        "test_inverse_metric_cg_factor_reuse_does_not_filter_zero_rows",
        "test_block_metric_schedule_must_match_representation",
        "test_block_inverse_metric_schedule_must_match_representation",
    ),
    "Composition reference checks run child operator anchors": (
        "test_composition_reference_check_runs_child_anchor",
        "test_composition_tune_writes_child_reference_rows",
    ),
    "Composition tests declare children through": (
        "test_typed_composition_records_sequential_combine_and_runtime_order",
        "test_public_tune_composition_uses_selected_child_rows",
        "test_composition_materialize_each_child_rejects_inline_child_lowering",
        "test_composition_validate_composed_output_skips_child_anchor",
    ),
    "Composition combinator tests cover `vp.compose`": (
        "test_public_tune_linear_combination_composition_executes_scaled_identity",
        "test_public_tune_source_composition_executes_batch_to_vector_child",
        "test_typed_composition_validates_combine_children_and_source_positions",
    ),
    "Positive-definiteness tests reject": (
        "test_inverse_metric_rejects_ill_conditioned_undamped_solve",
        "test_typed_inverse_metric_tol_lowers_to_matrix_free_cg",
        "test_matrix_free_inverse_metric_rows_admit_only_conjugate_gradient",
        "test_public_matrix_free_metric_tunes_selected_curvature_product",
    ),
    "EKFAC metric tests cover": (
        "test_typed_ekfac_metric_products_execute_against_reference",
        "test_ekfac_inverse_metric_factorized_solve_matches_dense_reference",
    ),
    "Square-root tests cover": (
        "test_sqrt_metric_cholesky_paths_match_reference",
        "test_ekfac_closed_form_square_root_paths_match_reference",
        "test_public_matrix_free_metric_square_roots_tune_selected_curvature_product",
        "test_sqrt_metric_cholesky_factor_round_trip_matches_metric",
        "test_sqrt_metric_eigenbasis_double_apply_matches_metric_product",
        "test_block_metric_square_root_factor_round_trip_matches_metric",
        "test_typed_kfac_metric_products_execute_against_reference",
    ),
    "Metric inner-product tests cover": (
        "test_metric_inner_dense_paths_match_reference",
        "test_typed_ekfac_metric_inner_products_execute_against_reference",
        "test_metric_inner_norm_requires_sqrt_apply_reduce",
        "test_metric_inner_block_rhs_matches_single_column_gram",
    ),
    "Per-example gradient tests cover": (
        "test_typed_softmax_cross_entropy_per_example_gradient_matches_reference",
        "test_typed_softmax_cross_entropy_empirical_fisher_matches_reference",
        "test_per_example_gradient_stacked_and_blockwise_execute_declared_rows",
    ),
    "Typed damping tests cover": (
        "test_typed_block_metric_per_group_damping_executes_by_block",
        "test_typed_kfac_metric_per_group_damping_executes_by_parameter",
        "test_typed_kfac_pi_damping_uses_factored_shift",
        "test_typed_kfac_rejects_invalid_factor_declarations",
    ),
    "Solver-tolerance tests cover": (
        "test_inverse_metric_cg_stops_at_declared_tol",
        "test_inverse_metric_reference_check_uses_declared_tol_threshold",
    ),
    "Cohort-input tests pin": (
        "test_public_tune_multi_product_cohort_without_run_dir",
        "test_tune_run_selects_complete_dtype_cohort",
        "test_cohort_constraint_pins_vector_axes_and_rejects_disagreement",
    ),
    "Multi-RHS tests cover": (
        "test_inverse_metric_multi_rhs_controls_cholesky_rhs_batching",
        "test_public_bound_operator_compiles_vector_step_once",
    ),
    "Typed-object validation tests reject": (
        "test_typed_softmax_cross_entropy_rejects_invalid_fields",
        "test_typed_sample_source_validation",
        "test_typed_declared_psd_matrix_free_rejects_indefinite_matvec",
    ),
    "Replay tests cover the new identity fields": (
        "test_public_problem_autotune_and_operator_load_replay",
        "test_public_operator_load_rejects_stale_model_identity",
        "test_typed_inverse_metric_tol_lowers_to_matrix_free_cg",
        "test_replay_identity_fields_distinguish_declared_variants",
    ),
    "KFAC metric multiply, inverse, and inner product": (
        "test_typed_kfac_metric_products_execute_against_reference",
    ),
    "Metric and inverse-metric checks reject nonsymmetric": (
        "test_metric_reference_check_rejects_nonsymmetric_metric",
        "test_metric_reference_check_rejects_indefinite_metric",
        "test_inverse_metric_reference_check_rejects_indefinite_metric",
    ),
    "Threshold logic covers": (
        "test_threshold_logic",
        "test_standard_reference_check_honors_strict_thresholds",
        "test_standard_reference_check_applies_numeric_error_bound_fields",
    ),
    "Compile tests cover disabled eager rows": (
        "test_public_space_compile_component_generates_conditional_rows",
        "test_standard_runtime_runs_real_torch_compile_whole_operator",
        "test_standard_runtime_enables_compiled_autograd_for_backward_operator",
        "test_standard_runtime_warms_compile_cache",
        "test_run_candidate_records_measured_recompile_count",
        "test_standard_runtime_compiles_whole_operator",
        "test_tune_thorough_strategy_records_compile_horizon_scores",
        "test_standard_runtime_fullgraph_rejects_graph_breaks",
    ),
    "Attention tests cover every frontend listed": (
        "test_transformers_attention_axis_uses_core_and_adapter_admission_fields",
        "test_sdpa_kernel_values_enter_declared_context",
        "test_sdpa_priority_list_enters_priority_context",
        "test_transformers_registered_attention_row_selects_runtime_backend",
        "test_transformers_operation_factory_sets_attention_and_runs_module",
    ),
    "Attention executor tests cover a non-Transformers module": (
        "test_mapping_attention_location_executes_non_transformers_attention",
        "test_pytorch_sdpa_direct_matches_exact_attention",
        "test_packed_exact_attention_restores_token_order",
        "test_blockwise_exact_attention_matches_full_attention",
        "test_packed_exact_attention_restores_padded_positions",
    ),
    "Distributed tests cover every distributed axis": (
        "test_distributed_adapter_registry_admits_owned_axes_and_strategy_fields",
        "test_distributed_operation_factory_applies_strategy_and_runs_module",
        "test_distributed_reference_check_uses_single_device_anchor",
        "test_reduce_rank_statuses_records_global_failure",
        "test_gloo_process_group_all_gather_matches_logical_rank_output",
        "test_nccl_process_group_single_rank_all_gather_matches_logical_rank_output",
        "test_nccl_process_group_all_gather_matches_logical_rank_output",
    ),
    "Search tests cover": (
        "test_tune_admission_strategy_returns_candidate_table_only",
        "test_tune_smoke_strategy_measures_baseline_and_class_c_rows",
        "test_tune_fast_strategy_compiles_near_fastest_eager_rows",
        "test_tune_balanced_strategy_crosses_retained_group_winners",
        "test_tune_thorough_strategy_uses_declared_repeat_count",
        "test_tune_exhaustive_strategy_measures_every_admitted_row",
    ),
    "Autobatch tests cover": (
        "test_autobatch_bridge_selects_candidate_by_positive_index_domain",
        "test_tune_fast_strategy_delegates_autobatch_domain_to_autobatch_find",
        "test_autobatch_domain_filters_reference_failures_before_probe",
        "test_plan_replay_preserves_autobatch_selected_value",
    ),
    "JSON schema validation rejects stale direct identity fields": (
        "test_record_current_rejects_stale_candidate_and_full_size_status",
        "test_check_record_current_rejects_stale_status_and_thresholds",
        "test_plan_replay_rejects_stale_context_and_materializer",
    ),
    "Saved reference and full-size rows carry schema-valid": (
        "test_tune_writes_admission_failure_rows_without_measurement",
        "test_plan_replay_recomputes_family_selection",
    ),
    "Saved-run replay materializes the selected plan": (
        "test_plan_replay_recomputes_family_selection",
        "test_public_problem_autotune_and_operator_load_replay",
    ),
    "Memory stability rejects post-call reserved growth": ("test_memory_stability",),
    "Measurement records required memory fields": (
        "test_measurement_timing_policy",
        "test_tune_writes_admission_failure_rows_without_measurement",
        "test_measurement_cleans_memory_backend_after_runtime_failure",
        "test_measurement_reuses_long_probe_as_measured_sample",
    ),
    "Selection chooses lower memory within": (
        "test_within_family_selection",
        "test_cohort_selection_prefers_lower_memory_near_fastest",
    ),
    "Selection tests cover cohort comparison": (
        "test_cohort_selection_sums_compiled_row_scores",
        "test_selection_scores_compiled_distributed_rows_by_global_compile_fields",
        "test_selection_tie_breaks_with_declared_rank_memory_reduction",
    ),
    "Dtype coherence is expressed through a": (
        "test_tune_run_selects_complete_dtype_cohort",
    ),
    "Cohort selection supports generic single-key": (
        "test_tune_run_uses_generic_multi_key_cohort_constraint",
        "test_tune_run_cohort_subset_handles_cross_boundary_dependencies",
    ),
    "Blocked descendants write": (
        "test_tune_run_propagates_candidate_validation_errors_inside_cohort",
        "test_tune_run_cohort_subset_handles_cross_boundary_dependencies",
    ),
    "Failed rows from non-selected cohort assignments": (
        "test_tune_run_cohort_subset_handles_cross_boundary_dependencies",
    ),
    "Plan replay rejects stale target": (
        "test_plan_replay_rejects_stale_context_and_materializer",
        "test_plan_replay_rejects_changed_memory_backend_identity",
    ),
    "Selection rejects stale signatures": (
        "test_selection_rejects_invalid_rows",
        "test_selection_requires_full_size_agreement",
    ),
    "`functional_call` tests cover": (
        "test_standard_runtime_executes_explicit_functional_call_settings",
        "test_standard_runtime_preserves_tied_parameter_aliases_during_dtype_cast",
        "test_standard_runtime_rejects_forbidden_functional_buffer_mutation",
        "test_module_functional_call_preserves_active_parametrization",
        "test_module_functional_call_handles_buffers_and_restores_mode",
        "test_module_functional_call_respects_tied_weight_policy",
    ),
    "`torch.func` tests cover": (
        "test_standard_axis_registry_validates_core_axes",
        "test_torch_func_admission_rejects_transform_limitations",
        "test_torch_func_admission_rejects_forward_ad_coverage_failure",
        "test_standard_operation_factory_runs_core_derivative_products",
        "test_torch_func_admission_accepts_declared_vmap_randomness",
        "test_torch_func_admission_rejects_invalid_vmap_randomness",
    ),
    "Checkpoint tests cover": (
        "test_checkpoint_operation_preserves_rng_state",
        "test_standard_runtime_executes_selective_checkpoint_with_context_pair",
    ),
    "Transformer adapter tests cover": (
        "test_transformers_operation_factory_sets_attention_and_runs_module",
        "test_transformers_sdpa_rows_enter_declared_kernel_context",
        "test_transformers_registered_attention_row_selects_runtime_backend",
        "test_transformers_model_identity_changes_with_replay_inputs",
        "test_transformers_runtime_rejects_unlowered_row_settings",
        "test_transformers_attention_location_executes_core_attention",
        "test_transformers_runtime_config_runs_full_size_check_and_materializer",
    ),
    "Distributed adapter tests cover": (
        "test_distributed_selected_settings_must_match_across_ranks",
        "test_distributed_record_contains_memory_surface_and_settings",
        "test_fsdp2_admission_requires_hook_entry_and_rejects_bypass",
        "test_layout_admission_uses_mode_specific_fields",
        "test_distributed_adapter_registry_admits_owned_axes_and_strategy_fields",
        "test_reduce_rank_statuses_records_global_failure",
        "test_distributed_operation_factory_delegates_dtensor_layout_to_strategy",
    ),
    "Selected-plan validation follows stored family order": (
        "test_validate_plan_materializes_in_validation_order",
        "test_tune_run_executes_declared_selected_plan_validators",
        "test_selected_plan_validation_writes_failed_record",
    ),
    "Root imports expose the user-facing surface": (
        "test_root_import_surface_exposes_front_door_and_hides_extensions",
    ),
    "Each pilot family can be expressed": (
        "test_pilot_lower_validates_family_problem_match",
    ),
    "Pilot selected settings can be produced": (
        "test_pilot_readiness_and_selected_settings",
    ),
    "Pilot selected-plan validation passes": (
        "test_pilot_selected_settings_require_validation_rows_when_plan_requires_them",
    ),
    "Downstream pilot stages accept": (
        "test_pilot_lowered_run_feeds_downstream_readiness_consumer",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def axis_manifest_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(SPEC_MANIFEST_START)
    end = text.index(SPEC_MANIFEST_END, start)

    return text[start:end]


def feature_axis_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("## Operator-Owned Axes")
    end = text.index("## Search Space Factoring", start)

    return text[start:end]


def acceptance_bullets_from_spec(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    start = text.index(SPEC_ACCEPTANCE_START)
    end = text.index(SPEC_ACCEPTANCE_END, start)

    return tuple(
        line[2:] for line in text[start:end].splitlines() if line.startswith("- ")
    )


def is_manifest_axis_key(key: str) -> bool:
    return ("." in key and "=" not in key) or key in MANIFEST_AXIS_KEY_ALLOWLIST


def axis_keys_from_markdown(source: str) -> tuple[str, ...]:
    keys = []

    for line in source.splitlines():
        match = MANIFEST_AXIS_BULLET.match(line)

        if match is None:
            continue

        key = match.group(1)

        if is_manifest_axis_key(key):
            keys.append(key)

    return tuple(dict.fromkeys(keys))


def axis_domains_from_spec(path: Path) -> Mapping[str, tuple[object, ...]]:
    domains = {}

    for line in axis_manifest_section(path).splitlines():
        match = MANIFEST_AXIS_BULLET.match(line)

        if match is None:
            continue

        key = match.group(1)

        if is_manifest_axis_key(key):
            domains[key] = spec_axis_domain(key, match.group(2) or "")

    return domains


def spec_axis_domain(key: str, tail: str) -> tuple[object, ...]:
    symbolic_domain = SPEC_SYMBOLIC_AXIS_DOMAINS.get(key)

    if symbolic_domain is not None:
        domain = symbolic_domain
    else:
        values = tuple(
            spec_axis_value(value) for value in re.findall(r"`([^`]+)`", tail)
        )

        if values:
            domain = values
        elif "positive integer tuple" in tail:
            domain = candidates_module.INTEGER_TUPLE_DOMAIN
        elif "positive integer" in tail:
            domain = candidates_module.INTEGER_DOMAIN
        elif "registered" in tail:
            domain = candidates_module.REGISTERED_DOMAIN
        elif "declared" in tail:
            domain = candidates_module.DECLARED_DOMAIN
        else:
            domain = candidates_module.DECLARED_DOMAIN

    return domain


def spec_axis_value(value: str) -> object:
    if value == "None":
        return None

    return value


def source_text_for_tests() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo_root() / "tests").glob("test_*.py"))
    )


def behavior_source_text_for_tests() -> str:
    """Return the joined source of test function bodies only.

    Module-level text such as the coverage mapping dicts is excluded, so a
    manifest value literal counts as covered only when it appears inside an
    executable test function.
    """
    segments = []

    for path in sorted((repo_root() / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = node.lineno

                if node.decorator_list:
                    start = min(d.lineno for d in node.decorator_list)

                segments.append("".join(lines[start - 1 : node.end_lineno]))

    return "".join(segments)


def collected_test_function_nodes() -> Mapping[str, ast.FunctionDef]:
    nodes = {}

    for path in sorted((repo_root() / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                nodes[node.name] = node

    return nodes


def has_behavioral_assertion(node: ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True

        if isinstance(sub, ast.Call):
            func = sub.func

            if isinstance(func, ast.Name) and func.id.startswith("assert"):
                return True

            if isinstance(func, ast.Attribute):
                if func.attr.startswith("assert"):
                    return True

                if func.attr == "raises" and any(
                    keyword.arg == "match" for keyword in sub.keywords
                ):
                    return True

    return False


def suite_test_names() -> set[str]:
    source = source_text_for_tests()

    return set(re.findall(r"def (test_[a-zA-Z0-9_]+)\(", source))


def manifest_value_literal_covered(value: object, source: str) -> bool:
    return str(value) in source


def assert_named_tests_exist(
    names: Sequence[str],
    existing_names: set[str],
) -> None:
    assert names
    assert set(names).issubset(existing_names)


class OneBatchData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "one_batch"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name, "source": "reference"}

    @staticmethod
    def probe_batches(family: str) -> tuple[Mapping[str, object], ...]:
        return ({"family": family, "source": "probe"},)


class OneVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "one_vector"}

    @staticmethod
    def reference_vectors(family: str) -> torch.Tensor:
        assert family

        return torch.tensor([1.0])

    @staticmethod
    def probe_vectors(family: str) -> tuple[torch.Tensor, ...]:
        assert family

        return (torch.tensor([1.0]),)


class TwoProbeData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_probe"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name, "source": "reference"}

    @staticmethod
    def probe_batches(family: str) -> tuple[Mapping[str, object], ...]:
        return (
            {"family": family, "source": "probe", "index": 0},
            {"family": family, "source": "probe", "index": 1},
        )


class TwoVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_vector"}

    @staticmethod
    def reference_vectors(family: str) -> torch.Tensor:
        assert family

        return torch.tensor([1.0])

    @staticmethod
    def probe_vectors(family: str) -> tuple[torch.Tensor, ...]:
        assert family

        return (torch.tensor([1.0]), torch.tensor([2.0]))


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1

        return value


def reference_passed() -> vpx.ReferenceResult:
    return vpx.ReferenceResult(
        "tree_close",
        {"max_abs_diff": 1e-6},
        {"max_abs_diff": 0.0},
    )


def cpu_target(timing_policy: vpx.TimingPolicy | None = None) -> vpx.Target:
    policy = vpx.TimingPolicy() if timing_policy is None else timing_policy

    return vpx.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float64", "fp32", "bf16", "fp16"),
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=policy,
        selection_policy=vpx.SelectionPolicy(),
        search_policy=vpx.SearchPolicy(strategy="exhaustive"),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def _input_signature(case: str = "test", family: str = "family") -> dict[str, object]:
    operator = ops.gradient(family, f"{case}-objective", aggregation="sum").signature()

    return {
        "case": case,
        "operator": operator,
        "operator_spec_hash": stable_hash(operator),
        "target": {"target": "test", "environment": {}},
        "adapter": {"adapter_id": "tests", "adapter_version": "1"},
    }


def test_environment_signature_captures_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    signature = vpx.environment_signature()

    assert signature["torch"]["version"] == torch.__version__
    assert isinstance(signature["torch"]["config"], str)
    assert signature["determinism"] == {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
    }
    assert signature["backend_flags"]["matmul_precision"] == (
        torch.get_float32_matmul_precision()
    )
    assert signature["backend_flags"]["cuda_matmul_allow_tf32"] is (
        torch.backends.cuda.matmul.allow_tf32
    )
    assert signature["env"]["PYTORCH_CUDA_ALLOC_CONF"] == ("expandable_segments:True")
    assert set(signature) == {
        "python",
        "platform",
        "torch",
        "determinism",
        "backend_flags",
        "cuda",
        "rocm",
        "mps",
        "env",
    }


def test_cuda_driver_version_is_none_when_runtime_does_not_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CUDARTWithoutDriverVersion:
        pass

    def fake_cudart() -> CUDARTWithoutDriverVersion:
        return CUDARTWithoutDriverVersion()

    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        fake_cudart,
    )

    assert cuda_driver_version() is None


def test_target_signature_includes_declared_device_identity() -> None:
    signature = cpu_target().signature()

    assert signature["device_signatures"] == (
        {
            "device": "cpu",
            "type": "cpu",
            "index": None,
        },
    )
    assert signature["search_policy"] == {
        "strategy": "exhaustive",
        "retained_top_count": None,
        "compile_call_horizons": (),
        "variance_repeat_count": None,
    }


def test_search_policy_rejects_unknown_strategy() -> None:
    with pytest.raises(RuntimeError, match="unsupported search strategy"):
        vpx.SearchPolicy(strategy="random")


def test_search_policy_requires_balanced_top_count() -> None:
    with pytest.raises(RuntimeError, match="retained_top_count"):
        vpx.SearchPolicy(strategy="balanced")

    with pytest.raises(RuntimeError, match="retained_top_count"):
        vpx.SearchPolicy(strategy="balanced", retained_top_count=True)

    with pytest.raises(RuntimeError, match="retained_top_count"):
        vpx.SearchPolicy(
            strategy="balanced",
            retained_top_count=unchecked_timing_policy_value(1.5),
        )


def test_search_policy_requires_thorough_fields() -> None:
    with pytest.raises(RuntimeError, match="compile_call_horizons"):
        vpx.SearchPolicy(strategy="thorough", retained_top_count=1)

    with pytest.raises(RuntimeError, match="variance_repeat_count"):
        vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(1,),
        )

    with pytest.raises(RuntimeError, match="compile_call_horizons"):
        vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(True,),
            variance_repeat_count=2,
        )

    with pytest.raises(RuntimeError, match="variance_repeat_count"):
        vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(1,),
            variance_repeat_count=True,
        )


def unchecked_timing_policy_value(value: Any) -> Any:
    return value


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: vpx.TimingPolicy(short_seconds=-1.0), "short_seconds"),
        (lambda: vpx.TimingPolicy(short_seconds=float("inf")), "short_seconds"),
        (lambda: vpx.TimingPolicy(short_seconds=True), "short_seconds"),
        (
            lambda: vpx.TimingPolicy(short_seconds=2.0, medium_seconds=1.0),
            "medium_seconds",
        ),
        (lambda: vpx.TimingPolicy(short_warmups=-1), "short_warmups"),
        (lambda: vpx.TimingPolicy(medium_warmups=True), "medium_warmups"),
        (
            lambda: vpx.TimingPolicy(long_warmups=unchecked_timing_policy_value(1.5)),
            "long_warmups",
        ),
        (lambda: vpx.TimingPolicy(short_measured_calls=0), "short_measured_calls"),
        (
            lambda: vpx.TimingPolicy(medium_measured_calls=False),
            "medium_measured_calls",
        ),
        (lambda: vpx.TimingPolicy(long_measured_calls=-1), "long_measured_calls"),
    ],
)
def test_timing_policy_rejects_invalid_counts_and_thresholds(
    factory: Callable[[], vpx.TimingPolicy],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        factory()


def test_timing_policy_allows_zero_thresholds_for_long_tier() -> None:
    policy = vpx.TimingPolicy(
        short_seconds=0.0,
        medium_seconds=0.0,
        long_warmups=0,
        long_measured_calls=1,
    )

    assert policy.plan(0.1) == (0, 1)


def test_tune_rejects_thorough_horizon_mismatch_before_probe() -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(),
            selection_policy=vpx.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(2,),
            variance_repeat_count=2,
        ),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-policy"},
        ),
    )

    with pytest.raises(vp.MaterializationError, match="selection horizon"):
        tune_problem(problem)

    assert calls == []


def test_tune_smoke_strategy_measures_baseline_and_class_c_rows(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidates = passed_changed_candidates(
        "family",
        (
            ("base", {}, ()),
            (
                "hvp-path",
                {"hvp.path": "reverse_over_reverse"},
                ("hvp.path",),
            ),
            (
                "gradient-graph",
                {"gradient.graph_schedule": "build_once"},
                ("gradient.graph_schedule",),
            ),
            ("dtype", {"dtype.model_compute": "fp32"}, ("dtype.model_compute",)),
            (
                "attention",
                {"attention.frontend": "transformers_sdpa"},
                ("attention.frontend",),
            ),
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="smoke"),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=TwoProbeData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-smoke"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp-path",
        "dtype",
    )
    assert tuple(
        row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)
    ) == (
        "attention",
        "base",
        "dtype",
        "gradient-graph",
        "hvp-path",
    )
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp-path",
        "hvp-path",
        "dtype",
        "dtype",
    )
    assert all(call[2]["index"] == 0 for call in calls if call[0] == "operation")
    assert plan.selected_candidate().candidate_id == "base"


def test_tune_smoke_strategy_requires_one_admitted_baseline() -> None:
    model = torch.nn.Linear(1, 1)
    target = dataclasses.replace(
        cpu_target(),
        search_policy=vpx.SearchPolicy(strategy="smoke"),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "hvp-path"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "hvp-path"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (
                vpx.Candidate(
                    "family",
                    "hvp-path",
                    {"hvp.path": "reverse_over_reverse"},
                    changed_axes=("hvp.path",),
                    admission_status="passed",
                ),
            ),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-smoke"},
        ),
    )

    with pytest.raises(vp.MaterializationError, match="baseline"):
        tune_problem(problem)


def test_tune_fast_strategy_compiles_near_fastest_eager_rows(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    compile_settings = {
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
    candidates = passed_changed_candidates(
        "family",
        (
            ("base", {}, ()),
            ("hvp", {"hvp.path": "reverse_over_reverse"}, ("hvp.path",)),
            ("input", {"input.residency": "gpu"}, ("input.residency",)),
            ("dtype", {"dtype.model_compute": "fp32"}, ("dtype.model_compute",)),
            (
                "compiled-hvp",
                {"hvp.path": "reverse_over_reverse", **compile_settings},
                ("hvp.path", "compile.enabled"),
            ),
            (
                "compiled-dtype",
                {"dtype.model_compute": "fp32", **compile_settings},
                ("dtype.model_compute", "compile.enabled"),
            ),
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="fast"),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vpx.Candidate,
            inputs: tuple[tuple[vpx.Batch, vpx.TensorTree], ...],
            output: vpx.TensorTree,
            samples: tuple[vpx.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.settings.get("compile.enabled") != "true":
                return {}

            return {
                "compile_time_seconds": 0.0,
                "steady_elapsed_seconds": 0.5,
                "recompile_count": 0,
            }

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-fast"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 3.0, 3.0, 5.0, 5.0, 6.0, 6.0, 7.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp",
        "dtype",
        "compiled-dtype",
    )
    assert tuple(
        row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)
    ) == (
        "base",
        "compiled-dtype",
        "compiled-hvp",
        "dtype",
        "hvp",
        "input",
    )
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp",
        "hvp",
        "dtype",
        "dtype",
        "compiled-dtype",
        "compiled-dtype",
    )
    assert plan.selected_candidate().candidate_id == "compiled-dtype"


def test_tune_balanced_strategy_crosses_retained_group_winners(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    compile_settings = {
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
    candidates = passed_changed_candidates(
        "family",
        (
            ("base", {}, ()),
            ("hvp-slow", {"hvp.path": "reverse_over_reverse"}, ("hvp.path",)),
            ("hvp-fast", {"hvp.path": "jvp_grad"}, ("hvp.path",)),
            ("dtype", {"dtype.model_compute": "fp32"}, ("dtype.model_compute",)),
            (
                "compiled-cross",
                {
                    "hvp.path": "jvp_grad",
                    "dtype.model_compute": "fp32",
                    **compile_settings,
                },
                ("hvp.path", "dtype.model_compute", "compile.enabled"),
            ),
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "balanced_compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vpx.Candidate,
            inputs: tuple[tuple[vpx.Batch, vpx.TensorTree], ...],
            output: vpx.TensorTree,
            samples: tuple[vpx.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.settings.get("compile.enabled") != "true":
                return {}

            return {
                "compile_time_seconds": 0.0,
                "steady_elapsed_seconds": 0.3,
                "recompile_count": 0,
            }

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            5.0,
            5.0,
            9.0,
            9.0,
            10.0,
            10.0,
            11.0,
            11.0,
            11.8,
            11.8,
            13.0,
        )),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
    )
    assert {row["candidate_id"] for row in saved_candidate_rows(tmp_path, plan)} == {
        "base",
        "hvp-slow",
        "hvp-fast",
        "dtype",
        "compiled-cross",
        "balanced:hvp-fast+dtype",
    }
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp-slow",
        "hvp-fast",
        "hvp-slow",
        "hvp-fast",
        "dtype",
        "dtype",
        "balanced:hvp-fast+dtype",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
        "compiled-cross",
    )
    assert plan.selected_candidate().candidate_id == "compiled-cross"

    saved_paths = tuple(
        sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json"))
    )
    second_plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock(()),
    )

    assert second_plan.selected_candidate().candidate_id == "compiled-cross"
    assert tuple(record.candidate_id for record in second_plan.full_size_records) == (
        "base",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
    )
    assert tuple(call[1] for call in calls) == (
        "base",
        "base",
        "hvp-slow",
        "hvp-fast",
        "hvp-slow",
        "hvp-fast",
        "dtype",
        "dtype",
        "balanced:hvp-fast+dtype",
        "balanced:hvp-fast+dtype",
        "compiled-cross",
        "compiled-cross",
    )
    assert (
        tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json")))
        == saved_paths
    )


def test_tune_balanced_strategy_halves_group_rows_by_probe_stage(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidates = passed_changed_candidates(
        "family",
        (
            ("base", {}, ()),
            ("hvp-slow", {"hvp.path": "reverse_over_reverse"}, ("hvp.path",)),
            ("hvp-middle", {"hvp.path": "jvp_grad"}, ("hvp.path",)),
            ("hvp-fast", {"hvp.path": "autograd_functional_hvp"}, ("hvp.path",)),
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert isinstance(vector, torch.Tensor)
        calls.append(("operation", candidate.candidate_id, batch["index"]))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls.append(("reference", candidate.candidate_id, None))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=TwoProbeData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced-halving"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            20.0,
            20.0,
            23.0,
            23.0,
            24.0,
            24.0,
            25.0,
            25.0,
            30.0,
            30.0,
            31.0,
        )),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base",
        "hvp-fast",
    )
    assert plan.selected_candidate().candidate_id == "hvp-fast"
    assert tuple(
        call for call in calls if call[0] == "operation" and call[1] != "base"
    ) == (
        ("operation", "hvp-slow", 0),
        ("operation", "hvp-middle", 0),
        ("operation", "hvp-fast", 0),
        ("operation", "hvp-fast", 1),
        ("operation", "hvp-middle", 1),
        ("operation", "hvp-fast", 0),
        ("operation", "hvp-fast", 1),
    )


def test_tune_balanced_strategy_with_only_baseline_measures_it_once(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "base", {}, admission_status="passed")
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="balanced", retained_top_count=1),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)
        calls.append(("operation", candidate.candidate_id))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls.append(("reference", candidate.candidate_id))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-balanced-baseline"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert tuple(record.candidate_id for record in plan.full_size_records) == ("base",)
    assert tuple(call[0] for call in calls) == ("reference", "operation")
    assert plan.selected_candidate().candidate_id == "base"


def test_tune_thorough_strategy_uses_declared_repeat_count(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "base", {}, admission_status="passed")
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(
                vpx.TimingPolicy(
                    short_seconds=0.0,
                    medium_seconds=0.0,
                    long_measured_calls=1,
                )
            ),
            selection_policy=vpx.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(3,),
            variance_repeat_count=2,
        ),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate.candidate_id, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate.candidate_id, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-thorough"},
        ),
    )
    clock = SequenceClock((0.0, 1.0, 1.0, 2.0, 2.0, 3.0))
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=clock,
    )

    assert plan.selected_candidate().candidate_id == "base"
    assert len(plan.records["family"].timing_samples) == 2
    assert tuple(call[1] for call in calls) == ("base", "base")
    assert clock.index == 6


def test_tune_thorough_strategy_records_compile_horizon_scores(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    compile_settings = {
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
    base = vpx.Candidate("family", "base", {}, admission_status="passed")
    compiled = vpx.Candidate(
        "family",
        "compiled",
        compile_settings,
        changed_axes=("compile.enabled",),
        admission_status="passed",
    )
    target = dataclasses.replace(
        dataclasses.replace(
            cpu_target(
                vpx.TimingPolicy(
                    short_seconds=0.0,
                    medium_seconds=0.0,
                    long_measured_calls=1,
                )
            ),
            selection_policy=vpx.SelectionPolicy(compile_call_horizon=3),
        ),
        search_policy=vpx.SearchPolicy(
            strategy="thorough",
            retained_top_count=1,
            compile_call_horizons=(3, 6),
            variance_repeat_count=2,
        ),
    )

    class CompileMetadataCheck:
        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"check": "thorough_compile_metadata"}

        @staticmethod
        def __call__(
            candidate: vpx.Candidate,
            inputs: tuple[tuple[vpx.Batch, vpx.TensorTree], ...],
            output: vpx.TensorTree,
            samples: tuple[vpx.Measurement, ...],
        ) -> Mapping[str, object]:
            assert inputs
            assert output is not None
            assert samples

            if candidate.candidate_id == "base":
                return {}

            assert candidate.candidate_id == "compiled"

            return {
                "compile_time_seconds": 6.0,
                "steady_elapsed_seconds": 2.0,
                "recompile_count": 1,
            }

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id in {"base", "compiled"}
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id in {"base", "compiled"}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (base, compiled),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-thorough-compile-horizons"},
            full_size_check=CompileMetadataCheck(),
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            20.0,
            20.0,
            30.0,
            30.0,
            31.0,
            31.0,
            32.0,
            32.0,
            33.0,
        )),
    )
    assert plan.selected_candidate().candidate_id == "compiled"
    metadata = plan.records["family"].selection_metadata
    scores = metadata["compile_amortized_seconds_by_horizon"]
    assert isinstance(scores, Mapping)
    assert scores["3"] == pytest.approx(6.0)
    assert scores["6"] == pytest.approx(4.0)

    saved_records, _ = saved_plan_rows(tmp_path, plan)
    saved_compiled = next(
        record for record in saved_records if record.candidate_id == "compiled"
    )
    saved_scores = saved_compiled.selection_metadata[
        "compile_amortized_seconds_by_horizon"
    ]
    assert isinstance(saved_scores, Mapping)
    assert saved_scores["3"] == pytest.approx(6.0)
    assert saved_scores["6"] == pytest.approx(4.0)


def test_tune_admission_strategy_returns_candidate_table_only(tmp_path: Path) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        "family",
        "row",
        {"dtype.model_compute": "fp32"},
        admission_status="passed",
    )
    target = dataclasses.replace(
        cpu_target(),
        search_policy=vpx.SearchPolicy(strategy="admission"),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        calls.append(("operation", candidate, batch, vector))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        calls.append(("reference", candidate, batch, vector))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.admission-search"},
        ),
    )
    plan = tune_problem(problem, run_dir=tmp_path)

    assert plan.selected == {}
    assert plan.records == {}
    assert plan.candidate_rows == (candidate,)
    assert plan.full_size_records == ()
    assert plan.check_records == ()
    assert calls == []
    assert saved_candidate_rows(tmp_path, plan)[0]["candidate_id"] == "row"
    assert not (tmp_path / "references").exists()
    assert not (tmp_path / "full_size").exists()

    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=(),
        check_records=(),
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={},
    )

    assert replayed.selected == {}
    assert tuple(candidate.signature() for candidate in replayed.candidate_rows) == (
        candidate.signature(),
    )


def test_extension_tensor_and_measurement_helpers() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.sum().backward()
    tree = {"value": torch.tensor([3.0, 4.0])}
    zeros = vpx.tree_zeros_like(tree)
    summed = vpx.tree_add(tree, zeros)

    vpx.clear_parameter_gradients((parameter,))

    assert parameter.grad is None
    assert vpx.tensor_signature(parameter)["requires_grad"] is True
    assert torch.equal(tree_leaves(zeros)[0], torch.zeros(2))
    assert torch.equal(tree_leaves(summed)[0], tree["value"])
    assert float(vpx.tree_l2_norm(tree)) == pytest.approx(5.0)


def test_tensor_tree_foreach_helpers_match_python_ops() -> None:
    left = {
        "a": torch.tensor([1.0, -2.0], dtype=torch.float64),
        "b": torch.tensor([3.0], dtype=torch.float64),
    }
    right = {
        "a": torch.tensor([4.0, 5.0], dtype=torch.float64),
        "b": torch.tensor([-6.0], dtype=torch.float64),
    }
    thresholds = {"max_abs_diff": 1e-12, "max_rel_diff": 1e-12}

    assert_tree_close(
        tree_add_foreach(left, right),
        {
            "a": torch.tensor([5.0, 3.0], dtype=torch.float64),
            "b": torch.tensor([-3.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_mul_foreach(left, 2.0),
        {
            "a": torch.tensor([2.0, -4.0], dtype=torch.float64),
            "b": torch.tensor([6.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_elementwise_mul_foreach(left, right),
        {
            "a": torch.tensor([4.0, -10.0], dtype=torch.float64),
            "b": torch.tensor([-18.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )
    assert_tree_close(
        tree_elementwise_div_foreach(right, left),
        {
            "a": torch.tensor([4.0, -2.5], dtype=torch.float64),
            "b": torch.tensor([-2.0], dtype=torch.float64),
        },
        thresholds=thresholds,
    )

    assert float(tree_dot_foreach(left, right)) == pytest.approx(-24.0)

    with pytest.raises(RuntimeError, match="same dtype"):
        tree_add_foreach(left, {"a": right["a"].float(), "b": right["b"]})


def test_axis_manifest_adapter_axes_use_adapter_admission_rules() -> None:
    axes = vpx.axis_manifest().by_key()

    attention_passed, attention_message = axes["attention.partition"].admit(
        vpx.Candidate(
            "attention",
            "missing-frontend",
            {"attention.partition": "full"},
        )
    )
    distributed_passed, distributed_message = axes["distributed.launch"].admit(
        vpx.Candidate(
            "distributed",
            "missing-strategy",
            {"distributed.launch": "torchrun"},
        )
    )
    registered_passed, registered_message = axes["attention.custom_kernel_id"].admit(
        vpx.Candidate(
            "attention",
            "empty-registered-id",
            {"attention.custom_kernel_id": ""},
        )
    )

    assert not attention_passed
    assert attention_message == "attention.partition requires attention.frontend"
    assert not distributed_passed
    assert distributed_message == "distributed.launch requires distributed.strategy"
    assert not registered_passed
    assert registered_message == (
        "candidate axis requires a registered id: attention.custom_kernel_id"
    )


def test_axis_manifest_carries_lowering_and_check_identity_fields() -> None:
    manifest = vpx.axis_manifest()

    for axis in manifest.axes:
        signature = axis.signature()
        setting_keys = set(axis.settings_keys)

        assert axis.admission_rule_id
        assert axis.lowering_rule_id
        assert axis.alias_normalization_rule
        assert axis.settings_keys_written == axis.settings_keys
        assert setting_keys.issubset(set(axis.admission_settings_keys))
        assert isinstance(axis.required_reference_checks, tuple)
        assert isinstance(axis.required_full_size_checks, tuple)
        assert signature["admission_rule_id"] == axis.admission_rule_id
        assert signature["lowering_rule_id"] == axis.lowering_rule_id
        assert signature["required_reference_checks"] == axis.required_reference_checks
        assert signature["required_full_size_checks"] == axis.required_full_size_checks
        assert signature["admission_settings_keys"] == axis.admission_settings_keys
        assert signature["settings_keys_written"] == axis.settings_keys_written

        if axis.adapter_id:
            assert axis.lowering_rule_id.startswith(axis.adapter_id)
        else:
            assert axis.lowering_rule_id.startswith("vptune.standard_runtime:")


def test_axis_manifest_matches_spec_key_and_value_domains() -> None:
    expected_domains = axis_domains_from_spec(repo_root() / "SPEC.md")
    actual_domains = {
        key: axis.value_domain for key, axis in vpx.axis_manifest().by_key().items()
    }

    assert actual_domains == expected_domains


def test_axis_manifest_keys_match_features_and_spec() -> None:
    spec_keys = tuple(axis_domains_from_spec(repo_root() / "SPEC.md"))
    feature_keys = axis_keys_from_markdown(
        feature_axis_source(repo_root() / "FEATURES.md")
    )

    assert set(spec_keys) == set(feature_keys)


def test_axis_manifest_values_and_checks_have_test_coverage() -> None:
    source = behavior_source_text_for_tests()
    existing_names = suite_test_names()
    value_gaps = []

    for axis in vpx.axis_manifest().axes:
        for value in axis.value_domain:
            if manifest_value_literal_covered(value, source):
                continue

            key = (axis.name, value)
            test_names = MANIFEST_SYMBOLIC_VALUE_TEST_COVERAGE.get(key)

            if test_names is None:
                value_gaps.append(key)
                continue

            assert_named_tests_exist(test_names, existing_names)

    required_checks = {
        check_id
        for axis in vpx.axis_manifest().axes
        for check_id in (
            *axis.required_reference_checks,
            *axis.required_full_size_checks,
        )
    }

    assert value_gaps == []
    assert set(MANIFEST_CHECK_TEST_COVERAGE) == required_checks

    for test_names in MANIFEST_CHECK_TEST_COVERAGE.values():
        assert_named_tests_exist(test_names, existing_names)


def test_spec_acceptance_tests_have_named_test_coverage() -> None:
    bullets = acceptance_bullets_from_spec(repo_root() / "SPEC.md")
    existing_names = suite_test_names()
    covered_prefixes = set()
    unmatched = []
    ambiguous = []

    for bullet in bullets:
        matches = tuple(
            prefix for prefix in ACCEPTANCE_TEST_COVERAGE if bullet.startswith(prefix)
        )

        if not matches:
            unmatched.append(bullet)
            continue

        if len(matches) > 1:
            ambiguous.append((bullet, matches))
            continue

        prefix = matches[0]
        covered_prefixes.add(prefix)
        assert_named_tests_exist(ACCEPTANCE_TEST_COVERAGE[prefix], existing_names)

    assert unmatched == []
    assert ambiguous == []
    assert covered_prefixes == set(ACCEPTANCE_TEST_COVERAGE)


@pytest.mark.parametrize(
    ("setting_key", "paired_settings"),
    [
        (
            "memory.primal_outputs",
            {
                "hvp.path": "reverse_over_reverse",
                "hvp.primal_reuse": "recompute_primal",
            },
        ),
        ("memory.jvp_outputs", {"ggn.jvp_reuse": "recompute_jvp"}),
        (
            "memory.output_cotangents",
            {"ggn.cotangent_reuse": "recompute_output_cotangent"},
        ),
    ],
)
def test_memory_output_recompute_admits_matching_lowering(
    setting_key: str,
    paired_settings: Mapping[str, object],
) -> None:
    registry = vpx.standard_axis_registry()
    candidate = vpx.Candidate(
        "hvp",
        "memory-output-recompute",
        {setting_key: "recompute", **paired_settings},
    )
    admitted = registry.admit(candidate)

    assert admitted.admission_status == "passed"


@pytest.mark.parametrize(
    "setting_key",
    [
        "memory.primal_outputs",
        "memory.jvp_outputs",
        "memory.output_cotangents",
    ],
)
def test_memory_output_recompute_rejects_unmatched_settings(
    setting_key: str,
) -> None:
    registry = vpx.standard_axis_registry()
    candidate = vpx.Candidate(
        "hvp",
        "memory-output-recompute",
        {setting_key: "recompute"},
    )
    admitted = registry.admit(candidate)

    assert admitted.admission_status == "failed"
    assert admitted.admission_error == (
        f"{setting_key}=recompute requires matching recompute settings"
    )


def test_memory_intermediate_residency_admits_only_owned_boundaries() -> None:
    registry = vpx.standard_axis_registry()
    torch_func_fields = {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": True,
        "forward_ad_supported": True,
    }
    ggn_candidate = vpx.Candidate(
        "ggn",
        "ggn-intermediate-residency",
        {
            "ggn.jvp_path": "torch_func_jvp",
            "memory.intermediate_residency": "cpu_staged",
            **torch_func_fields,
        },
    )
    composition_candidate = vpx.Candidate(
        "composition",
        "composition-intermediate-residency",
        {
            "composition.execution": "stream_child_outputs",
            "memory.intermediate_residency": "cpu_staged",
        },
    )
    hvp_candidate = vpx.Candidate(
        "hvp",
        "hvp-intermediate-residency",
        {
            "hvp.path": "reverse_over_reverse",
            "memory.intermediate_residency": "cpu_staged",
        },
    )
    fused_composition_candidate = vpx.Candidate(
        "composition",
        "fused-composition-intermediate-residency",
        {
            "composition.execution": "fuse_adjacent_children",
            "memory.intermediate_residency": "cpu_staged",
        },
    )

    assert registry.admit(ggn_candidate).admission_status == "passed"
    assert registry.admit(composition_candidate).admission_status == "passed"
    assert registry.admit(hvp_candidate).admission_error == (
        "memory.intermediate_residency requires named intermediate boundaries"
    )
    assert registry.admit(fused_composition_candidate).admission_error == (
        "memory.intermediate_residency requires visible composition child boundaries"
    )


def test_matrix_free_lanczos_requires_declared_iterations() -> None:
    registry = vpx.standard_axis_registry()
    matrix_free_candidate = vpx.Candidate(
        "sqrt_metric",
        "matrix-free-lanczos",
        {
            "sqrt_metric.factor_path": "matrix_free_lanczos",
            "sqrt_metric.lanczos_iterations": 8,
        },
    )
    stray_iterations = vpx.Candidate(
        "sqrt_metric",
        "stray-lanczos-iterations",
        {"sqrt_metric.lanczos_iterations": 8},
    )

    assert registry.admit(matrix_free_candidate).admission_status == "passed"
    assert registry.admit(stray_iterations).admission_error == (
        "sqrt_metric.lanczos_iterations requires matrix_free_lanczos"
    )


def test_matrix_free_preconditioner_rejected_until_sibling_product_lowered() -> None:
    registry = vpx.standard_axis_registry()
    missing_product = vpx.Candidate(
        "inverse_metric",
        "matrix-free-preconditioner",
        {"inverse_metric.preconditioner": "matrix_free"},
    )
    admitted = vpx.Candidate(
        "inverse_metric",
        "matrix-free-preconditioner",
        {
            "inverse_metric.preconditioner": "matrix_free",
            "inverse_metric.preconditioner_product": "preconditioner",
        },
    )
    stray_product = vpx.Candidate(
        "inverse_metric",
        "stray-preconditioner-product",
        {
            "inverse_metric.preconditioner": "none",
            "inverse_metric.preconditioner_product": "preconditioner",
        },
    )

    assert registry.admit(missing_product).admission_error == (
        "inverse_metric.preconditioner=matrix_free requires "
        "inverse_metric.preconditioner_product"
    )
    assert registry.admit(admitted).admission_status == "passed"
    assert registry.admit(stray_product).admission_error == (
        "inverse_metric.preconditioner_product applies only to matrix_free "
        "preconditioner"
    )


@pytest.mark.parametrize(
    ("setting_key", "value"),
    [
        ("layout.params", "per_shard"),
        ("layout.params", "dtensor"),
        ("layout.vector", "per_shard"),
        ("layout.vector", "dtensor"),
        ("layout.output", "per_shard"),
        ("layout.output", "dtensor"),
    ],
)
def test_distributed_layout_values_require_distributed_adapter(
    setting_key: str,
    value: str,
) -> None:
    registry = vpx.standard_axis_registry()
    core_candidate = vpx.Candidate(
        "gradient",
        "core-distributed-layout",
        {setting_key: value},
    )
    combined_registry = vpx.standard_axis_registry()
    combined_registry.register(
        vpx.AxisDescriptor(
            "distributed.strategy",
            ("distributed.strategy",),
            ("tensor_parallel",),
        )
    )
    distributed_candidate = vpx.Candidate(
        "gradient",
        "adapter-distributed-layout",
        {setting_key: value, "distributed.strategy": "tensor_parallel"},
    )

    assert registry.admit(core_candidate).admission_error == (
        f"{setting_key}={value} requires distributed adapter ownership"
    )
    assert combined_registry.admit(distributed_candidate).admission_status == "passed"


def test_standard_compile_boundary_rejects_transformer_block_without_adapter() -> None:
    registry = vpx.standard_axis_registry()
    candidate = vpx.Candidate(
        "gradient",
        "plain-transformer-block",
        {
            "gradient.path": "torch_autograd_grad",
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
    )

    assert registry.admit(candidate).admission_error == (
        "compile.boundary=transformer_block is not lowered for gradient"
    )


def valid_sampling_bound() -> dict[str, object]:
    return {
        "kind": "abs_or_rel",
        "max_abs_diff": 1e-12,
        "max_rel_diff": 1e-12,
        "norm_floor": 1e-12,
    }


def reduction_bound_fields() -> dict[str, object]:
    return {
        "k": 2,
        "epsilon": 0.01,
        "C_op": 1.5,
        "S_row": 3.0,
        "output_norm_floor": 1e-6,
    }


def test_cohort_constraint_rejects_unsupported_modes() -> None:
    with pytest.raises(RuntimeError, match="dependency inheritance"):
        vpx.CohortConstraint(
            name="bad",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
            dependency_inheritance="all_families",
        )

    with pytest.raises(RuntimeError, match="selection aggregation"):
        vpx.CohortConstraint(
            name="bad",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
            selection_aggregation="mean_elapsed_seconds",
        )


def materialize_candidate_impl(
    candidate: vpx.Candidate,
    record: vpx.FullSizeRecord,
) -> vpx.CandidateOperation:
    def operation() -> torch.Tensor:
        return torch.tensor([float(candidate.settings.get("scale", 1.0))])

    assert record.candidate_id == candidate.candidate_id

    return operation


materialize_candidate = vpx.CallableMaterializer(
    "tests.materialize_candidate",
    "1",
    {},
    {"callback": "tests.materialize_candidate_impl"},
    materialize_candidate_impl,
)


def runtime_config(
    candidates: Sequence[vpx.Candidate],
    operation_factory: vpx.OperationFactoryCallback,
    reference_check: vpx.ReferenceCheckCallback,
    materializer: vpx.Materializer,
    axis_registry: vpx.CandidateAdmitter | None,
    signature: Mapping[str, Any],
    autobatch_domains: tuple[vpx.AutobatchDomain, ...] = (),
    *,
    full_size_check: vpx.FullSizeCheck | None = None,
    reference_check_name: str = "tree_close",
) -> vpx.RuntimeConfig:
    settings = {"runtime": dict(signature)}
    operation_factory = vpx.CallableOperationFactory(
        "tests.operation_factory",
        "1",
        settings,
        {"callback": "tests.operation_factory"},
        operation_factory,
    )
    reference_check = vpx.CallableReferenceCheck(
        "tests.reference_check",
        "1",
        settings,
        {"callback": "tests.reference_check"},
        reference_check,
    )

    return vpx.RuntimeConfig(
        tuple(candidates),
        operation_factory,
        reference_check,
        materializer,
        axis_registry,
        signature,
        autobatch_domains=autobatch_domains,
        full_size_check=full_size_check,
        reference_check_name=reference_check_name,
    )


def passed_candidate(
    family: str,
    candidate_id: str,
    settings: Mapping[str, Any],
) -> vpx.Candidate:
    return vpx.Candidate(
        family,
        candidate_id,
        settings,
        admission_status="passed",
    )


def passed_candidates(
    family: str,
    rows: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[vpx.Candidate, ...]:
    return tuple(
        passed_candidate(family, candidate_id, settings)
        for candidate_id, settings in rows
    )


def passed_changed_candidates(
    family: str,
    rows: Sequence[tuple[str, Mapping[str, Any], tuple[str, ...]]],
) -> tuple[vpx.Candidate, ...]:
    return tuple(
        vpx.Candidate(
            family,
            candidate_id,
            settings,
            changed_axes=changed_axes,
            admission_status="passed",
        )
        for candidate_id, settings, changed_axes in rows
    )


def reference_check_never_runs(
    candidate: vpx.Candidate,
    batch: Mapping[str, object],
    vector: vpx.TensorTree,
) -> vpx.ReferenceResult:
    _ = candidate, batch, vector
    message = "reference check should not run"
    raise AssertionError(message)


def operation_factory_never_runs(
    candidate: vpx.Candidate,
    batch: Mapping[str, object],
    vector: vpx.TensorTree,
) -> vpx.CandidateOperation:
    _ = candidate, batch, vector
    message = "operation should not run"
    raise AssertionError(message)


def constant_operation_factory(
    candidate: vpx.Candidate,
    batch: Mapping[str, object],
    vector: vpx.TensorTree,
) -> vpx.CandidateOperation:
    assert isinstance(candidate, vpx.Candidate)
    assert isinstance(batch, Mapping)

    return vpx.constant_operation(vector)


def passing_reference_check(
    candidate: vpx.Candidate,
    batch: Mapping[str, object],
    vector: vpx.TensorTree,
) -> vpx.ReferenceResult:
    assert isinstance(candidate, vpx.Candidate)
    assert isinstance(batch, Mapping)
    assert vector is not None

    return reference_passed()


def recorded_tuning_problem(
    *,
    model: torch.nn.Module,
    target: vpx.Target,
    name: str,
    operator: vpx.OperatorSpec,
    candidates: tuple[vpx.Candidate, ...],
    calls: list[str],
    failing_reference_candidate: str | None = None,
    generator: str | None = None,
    record_call: Callable[[vpx.Candidate], str] | None = None,
) -> vpx.Problem:
    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.family == name
        assert batch["family"] == name
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        if candidate.candidate_id == failing_reference_candidate:
            message = "reference failed"
            raise RuntimeError(message)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.family == name
        assert batch["family"] == name
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        def operation() -> torch.Tensor:
            if record_call is None:
                calls.append(candidate.candidate_id)
            else:
                calls.append(record_call(candidate))

            return vector

        return operation

    return vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": name if generator is None else generator},
        ),
    )


def one_call_cpu_target() -> vpx.Target:
    return cpu_target(
        vpx.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )


def single_row_tuning_problem(
    *,
    model: torch.nn.Module,
    target: vpx.Target,
    operator: vpx.OperatorSpec,
    row: vpx.Candidate,
    calls: dict[str, int],
    generator: str,
    reference_failure_call: int | None = None,
) -> vpx.Problem:
    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == row.candidate_id
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        calls["reference"] += 1

        if calls["reference"] == reference_failure_call:
            message = "reference rejected row"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == row.candidate_id
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls["operation"] += 1

            return vector

        return operation

    return vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (row,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": generator},
        ),
    )


def test_runtime_config_requires_identity_bearing_callbacks() -> None:
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch == {}

        return vpx.constant_operation(vector)

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch == {}
        assert vector is not None

        return reference_passed()

    operation_factory_with_identity = vpx.CallableOperationFactory(
        "tests.operation_factory",
        "1",
        {},
        {"callback": "tests.operation_factory"},
        operation_factory,
    )
    reference_check_with_identity = vpx.CallableReferenceCheck(
        "tests.reference_check",
        "1",
        {},
        {"callback": "tests.reference_check"},
        reference_check,
    )

    def untyped(value: Any) -> Any:
        return value

    with pytest.raises(vp.MaterializationError, match="operation_factory"):
        vpx.RuntimeConfig(
            (candidate,),
            untyped(operation_factory),
            reference_check_with_identity,
            materialize_candidate,
            None,
            {"runtime": "test.raw-callback"},
        )

    with pytest.raises(vp.MaterializationError, match="reference_check"):
        vpx.RuntimeConfig(
            (candidate,),
            operation_factory_with_identity,
            untyped(reference_check),
            materialize_candidate,
            None,
            {"runtime": "test.raw-callback"},
        )


def replay_context_for_plan(
    plan: vpx.Plan,
    *,
    validation_required: bool = False,
) -> vpx.ReplayContext:
    family_input_signatures = {
        family: record.input_signature for family, record in plan.records.items()
    }

    for record in plan.check_records:
        family_input_signatures.setdefault(record.family, record.input_signature)

    return vpx.ReplayContext(
        input_signature=plan.input_signature,
        family_input_signatures=family_input_signatures,
        materializer_identities=plan.materializer_identities(),
        selection_policy=plan.policy,
        target_identity=plan.target_identity,
        runtime_identities=plan.selected_runtime_identities(),
        adapter_identities=plan.selected_adapter_identities(),
        validator_identities=plan.selected_validator_identities(),
        validation_required=validation_required,
        validation_order=plan.validation_order,
    )


def saved_plan_rows(
    run_dir: Path,
    plan: vpx.Plan,
) -> tuple[tuple[vpx.FullSizeRecord, ...], tuple[vpx.CheckRecord, ...]]:
    saved_full_size_rows = tuple(
        vpx.full_size_record_from_json(read_record(path))
        for path in sorted((run_dir / "full_size").rglob("*.json"))
    )
    saved_check_rows = tuple(
        vpx.check_record_from_json(read_record(path))
        for path in sorted((run_dir / "references").rglob("*.json"))
    )
    full_size_by_key = {
        canonical_json(record.row_key()): record for record in saved_full_size_rows
    }
    check_by_key = {
        canonical_json(record.row_key()): record for record in saved_check_rows
    }
    full_size_rows = tuple(
        full_size_by_key[canonical_json(record.row_key())]
        for record in plan.full_size_records
    )
    check_rows = tuple(
        check_by_key[canonical_json(record.row_key())] for record in plan.check_records
    )

    return full_size_rows, check_rows


def saved_candidate_rows(
    run_dir: Path,
    _: vpx.Plan,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        read_record(path) for path in sorted((run_dir / "candidates").rglob("*.json"))
    )


def candidate_records_for_plan(plan: vpx.Plan) -> tuple[Mapping[str, object], ...]:
    def candidate_for_record(record: vpx.FullSizeRecord) -> vpx.Candidate:
        selected_candidate = plan.selected.get(record.family)

        if (
            selected_candidate is not None
            and selected_candidate.candidate_id == record.candidate_id
            and to_json_value(selected_candidate.settings)
            == to_json_value(record.candidate_settings)
        ):
            return selected_candidate

        return vpx.Candidate(
            family=record.family,
            candidate_id=record.candidate_id,
            settings=dict(record.candidate_settings),
            dependency_identities={
                family: dict(identity)
                for family, identity in record.dependency_identities.items()
            },
            cohort_assignment=dict(record.cohort_assignment),
            admission_status="passed",
            generator_id=record.generator_id,
            generator_version=record.generator_version,
        )

    return tuple(
        vpx.candidate_record_to_json(
            candidate_for_record(record), record.input_signature
        )
        for record in plan.full_size_records
    )


def test_candidate_record_round_trips_migration_source_id() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
        migration_source_id="pilot-row-1",
    )
    row = vpx.candidate_record_to_json(
        candidate,
        _input_signature("migration-source"),
    )
    replayed = vpx.candidate_record_from_json(row)

    assert row["migration_source_id"] == "pilot-row-1"
    assert replayed.migration_source_id == "pilot-row-1"
    assert replayed.signature() == candidate.signature()


def test_candidate_record_writes_declared_hook_ids(tmp_path: Path) -> None:
    candidate = vpx.Candidate(
        "family",
        "hooks",
        {
            "activation.offload": "custom_saved_tensor_hooks",
            "activation.pack_hook": "pack",
            "activation.unpack_hook": "unpack",
            "checkpoint.context_fn": "declared_context_pair",
            "checkpoint.context_fn_callable": "default",
        },
        admission_status="passed",
    )
    path = tmp_path / "candidate.json"

    write_record(path, vpx.candidate_record_to_json(candidate, _input_signature("ids")))
    loaded = read_record(path)

    assert loaded["candidate_settings"] == candidate.settings


def test_write_record_exclusive_preserves_existing_record(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    first = vpx.Candidate("family", "first", {}, admission_status="passed")
    second = vpx.Candidate("family", "second", {}, admission_status="passed")

    write_record(path, vpx.candidate_record_to_json(first, _input_signature("first")))

    with pytest.raises(FileExistsError):
        write_record_exclusive(
            path,
            vpx.candidate_record_to_json(second, _input_signature("second")),
        )

    assert read_record(path)["candidate_id"] == "first"


def test_write_unique_record_moves_to_numbered_sibling(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    first = vpx.Candidate("family", "first", {}, admission_status="passed")
    second = vpx.Candidate("family", "second", {}, admission_status="passed")

    run_module._write_unique_record(
        path,
        vpx.candidate_record_to_json(first, _input_signature("first")),
    )
    run_module._write_unique_record(
        path,
        vpx.candidate_record_to_json(second, _input_signature("second")),
    )

    assert read_record(path)["candidate_id"] == "first"
    assert read_record(tmp_path / "candidate-000001.json")["candidate_id"] == "second"


def test_write_record_rejects_type_specific_missing_fields(tmp_path: Path) -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    row = vpx.candidate_record_to_json(candidate, _input_signature("schema"))
    missing_status = dict(row)
    missing_status.pop("status")

    with pytest.raises(vp.RecordFormatError, match="status"):
        write_record(tmp_path / "missing-status.json", missing_status)

    summary = {
        "record_type": "summary",
        "schema_version": row["schema_version"],
        "package_version": row["package_version"],
        "input_signature": {},
        "candidate_settings": {},
        "status": "passed",
        "generator_id": "plan",
        "generator_version": row["package_version"],
        "selected": {},
        "candidate_rows": (),
        "records": {},
        "full_size_records": (),
        "check_records": (),
        "validation_records": (),
        "validation_required": False,
        "validation_order": (),
        "validator_identities": {},
        "dependencies_by_family": {},
        "cohort_assignment": None,
        "cohort_constraints": (),
        "selected_dependency_identities": {},
        "materializer_identities": {},
        "runtime_identities": {},
        "adapter_identities": {},
        "policy": dataclasses.asdict(vpx.SelectionPolicy()),
    }

    with pytest.raises(vp.RecordFormatError, match="target_identity"):
        write_record(tmp_path / "missing-target.json", summary)


def test_stable_hash_changes_on_identity_inputs() -> None:
    first = {
        "parameter_order": ("weight", "bias"),
        "data": {"slice": "a"},
        "target": {"device": "cpu"},
        "generator_version": "1",
    }
    second = {
        "parameter_order": ("bias", "weight"),
        "data": {"slice": "a"},
        "target": {"device": "cpu"},
        "generator_version": "1",
    }

    assert stable_hash(first) == stable_hash(dict(first))
    assert stable_hash(first) != stable_hash(second)


def test_module_identity_records_tied_parameters_and_devices() -> None:
    class TiedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.nn.Parameter(torch.ones(2))
            self.first = shared
            self.second = shared
            self.register_buffer("scale", torch.ones(1))

    identity = module_identity(TiedModule())

    assert identity["tied_parameter_groups"] == (("first", "second"),)
    assert identity["parameters"][0]["device"] == "cpu"
    assert identity["buffers"][0]["device"] == "cpu"

    preserved = vpx.parameter_surface(TiedModule(), tied_weights="preserve")
    deduplicated = vpx.parameter_surface(TiedModule(), tied_weights="deduplicate")

    assert preserved.names == ("first", "second")
    assert deduplicated.names == ("first",)

    with pytest.raises(RuntimeError):
        vpx.parameter_surface(TiedModule(), tied_weights="unknown")

    grouped = vpx.parameter_surface(
        TiedModule(),
        layer_groups=(("first", "second"),),
        block_groups=(("first", "second"),),
    )

    assert grouped.layer_groups == (("first", "second"),)
    assert grouped.block_groups == (("first", "second"),)
    assert grouped.signature()["layer_groups"] == (("first", "second"),)

    with pytest.raises(RuntimeError, match="layer_groups"):
        vpx.parameter_surface(TiedModule(), layer_groups=(("first",),))

    with pytest.raises(RuntimeError, match="parametrization_policy"):
        vpx.ParameterSurface(
            names=("first",),
            shapes=((2,),),
            trainable=(True,),
            parametrization_policy="disabled",
        )


def test_module_identity_records_nested_parametrizations() -> None:
    class ExpParametrization(torch.nn.Module):
        @override
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.exp()

    class NegParametrization(torch.nn.Module):
        @override
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return -value

    class NestedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(1, 1, bias=False)

    first = NestedModule()
    second = NestedModule()
    parametrize.register_parametrization(first.block, "weight", ExpParametrization())
    parametrize.register_parametrization(second.block, "weight", NegParametrization())

    first_identity = module_identity(first)
    second_identity = module_identity(second)

    assert first_identity["parametrizations"][0]["parameter"] == "block.weight"
    parametrization_name = first_identity["parametrizations"][0]["parametrizations"][0]
    assert parametrization_name.endswith("ExpParametrization")
    assert first_identity["parametrizations"] != second_identity["parametrizations"]


def test_threshold_logic() -> None:
    thresholds = thresholds_for_measurements(
        {"max_abs_diff": 1e-4, "max_rel_diff": 2.0},
        {"dtype.model_compute": "fp16"},
    )

    assert thresholds["max_abs_diff"] == pytest.approx(1e-4)
    validate_thresholds({"max_abs_diff": 1e-5, "max_rel_diff": 10.0}, thresholds)
    validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 1e-5}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 10.0}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": math.nan}, {"max_abs_diff": 1e-4})

    bound_measurements = numeric_error_bound_measurements(
        {"numeric.float32_matmul_precision": "high"},
        reduction_bound_fields(),
        torch.tensor([4.0]),
    )

    assert bound_measurements["numeric_error_bound_abs"] == pytest.approx(
        1.5 * (0.02 / 0.98) * 3.0
    )
    assert bound_measurements["numeric_error_bound_rel"] == pytest.approx(
        bound_measurements["numeric_error_bound_abs"] / 4.0
    )
    assert uses_reduction_degrading_setting({"fsdp.mp_policy.reduce_dtype": "bf16"})
    assert not uses_reduction_degrading_setting({"fsdp.mp_policy.reduce_dtype": "fp32"})
    validate_numeric_error_bound(
        {"max_abs_diff": 0.08, "max_rel_diff": 0.02},
        {"max_abs_diff": 0.1, "max_rel_diff": 0.1},
        bound_measurements,
    )

    with pytest.raises(ReferenceFailedError, match="exceeds derived bound"):
        validate_numeric_error_bound(
            {"max_abs_diff": 0.2, "max_rel_diff": 0.2},
            {"max_abs_diff": 0.5, "max_rel_diff": 0.5},
            bound_measurements,
        )

    with pytest.raises(ReferenceFailedError, match="bound exceeds threshold"):
        validate_numeric_error_bound(
            {"max_abs_diff": 0.01, "max_rel_diff": 0.01},
            {"max_abs_diff": 0.01, "max_rel_diff": 0.01},
            bound_measurements,
        )

    with pytest.raises(ReferenceFailedError, match="fields are missing"):
        numeric_error_bound_measurements(
            {"numeric.float32_matmul_precision": "high"},
            {},
            torch.tensor([4.0]),
        )

    bad_bound_fields = dict(reduction_bound_fields())
    bad_bound_fields["k"] = 100

    with pytest.raises(ReferenceFailedError, match=r"k [*] epsilon"):
        numeric_error_bound_measurements(
            {"numeric.float32_matmul_precision": "high"},
            bad_bound_fields,
            torch.tensor([4.0]),
        )

    assert_tree_close(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
    )

    with pytest.raises(ReferenceFailedError):
        assert_tree_close(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )

    with pytest.raises(RuntimeError):
        assert_tree_close(
            torch.zeros(1, 2),
            torch.zeros(2),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )

    with pytest.raises(ReferenceFailedError):
        assert_tree_close(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0 + 1e-5], dtype=torch.float32),
            settings={"dtype.model_compute": "fp16"},
            thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
        )


def test_gradient_jvp_vjp_hvp_anchors() -> None:
    params = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float64)
    vector = torch.tensor([0.7, -0.2, 0.1], dtype=torch.float64)
    cotangent = torch.tensor([1.3, -0.4, 0.8], dtype=torch.float64)

    def scalar(input_params: torch.Tensor) -> torch.Tensor:
        return input_params.pow(3).sum() + 0.5 * input_params.pow(2).sum()

    def function(input_params: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            input_params[0] * input_params[1],
            input_params[1].sin(),
            input_params[2].pow(2),
        ))

    gradient = vpx.gradient_anchor(scalar, params)
    expected_gradient = 3.0 * params.pow(2) + params

    assert torch.allclose(gradient, expected_gradient)
    assert torch.allclose(
        vpx.jvp_anchor(function, params, vector),
        vpx.finite_difference_jvp(function, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert torch.allclose(
        vpx.forward_ad_jvp_anchor(function, params, vector),
        vpx.jvp_anchor(function, params, vector),
        atol=1e-12,
    )
    assert vpx.vjp_dot_identity_error(function, params, vector, cotangent) < 1e-12
    assert torch.allclose(
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        vpx.finite_difference_hvp(scalar, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert torch.allclose(
        vpx.hvp_anchor(scalar, params, vector),
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        atol=1e-12,
    )
    assert torch.allclose(
        vpx.vhp_anchor(scalar, params, vector),
        vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        atol=1e-12,
    )


def test_tree_gradient_and_hvp_return_zero_for_disconnected_leaves() -> None:
    params = {
        "active": torch.tensor([2.0], dtype=torch.float64),
        "disconnected": torch.tensor([3.0], dtype=torch.float64),
    }
    vector = {
        "active": torch.tensor([0.5], dtype=torch.float64),
        "disconnected": torch.tensor([7.0], dtype=torch.float64),
    }

    def scalar(tree: dict[str, torch.Tensor]) -> torch.Tensor:
        return tree["active"].pow(2).sum()

    gradient = vpx.gradient_anchor(scalar, params)
    hvp = vpx.hvp_reverse_over_reverse_anchor(scalar, params, vector)

    assert isinstance(gradient, dict)
    assert isinstance(hvp, dict)
    assert torch.allclose(gradient["active"], torch.tensor([4.0], dtype=torch.float64))
    assert torch.allclose(
        gradient["disconnected"],
        torch.zeros(1, dtype=torch.float64),
    )
    assert torch.allclose(hvp["active"], torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(hvp["disconnected"], torch.zeros(1, dtype=torch.float64))


def test_forward_ad_jvp_anchor_supports_module_functional_call() -> None:
    module = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    params = {"weight": torch.tensor([[1.0, -2.0]], dtype=torch.float64)}
    vector = {"weight": torch.tensor([[0.5, 1.5]], dtype=torch.float64)}
    inputs = torch.tensor([[3.0, 4.0]], dtype=torch.float64)

    def function(active_params: dict[str, torch.Tensor]) -> torch.Tensor:
        output = vpx.module_functional_call(
            module,
            active_params,
            {},
            inputs,
            module_mode="eval",
            tie_weights=True,
            strict=False,
            parametrization_policy="active",
            mutates_state=False,
            mutated_parameter_keys=(),
            mutated_buffer_keys=(),
        )
        assert isinstance(output, torch.Tensor)

        return output

    result = vpx.forward_ad_jvp_anchor(function, params, vector)
    expected = inputs @ vector["weight"].T

    assert torch.allclose(result, expected)


def test_tensor_tree_preserves_mapping_insertion_order() -> None:
    tree = {
        "second": torch.tensor([2.0]),
        "first": torch.tensor([1.0]),
    }
    mapped = tree_map(lambda tensor: tensor + 1.0, tree)
    rebuilt = tree_from_leaves(
        tree,
        (torch.tensor([20.0]), torch.tensor([10.0])),
    )
    signature = tree_signature(tree)

    assert isinstance(mapped, dict)
    assert isinstance(rebuilt, dict)
    assert tuple(mapped) == ("second", "first")
    assert [float(leaf.item()) for leaf in tree_leaves(tree)] == [2.0, 1.0]
    assert torch.equal(tree_leaves(rebuilt)[0], torch.tensor([20.0]))
    assert torch.equal(tree_leaves(rebuilt)[1], torch.tensor([10.0]))
    assert signature["items"][0]["key"] == "second"
    assert signature["items"][1]["key"] == "first"

    with pytest.raises(RuntimeError, match="too few"):
        tree_from_leaves(tree, (torch.tensor([20.0]),))

    with pytest.raises(RuntimeError, match="too many"):
        tree_from_leaves(
            tree,
            (
                torch.tensor([20.0]),
                torch.tensor([10.0]),
                torch.tensor([30.0]),
            ),
        )


def test_dense_ggn_fisher_and_metric_anchors() -> None:
    params = torch.tensor([0.3, -0.2], dtype=torch.float64)
    vector = torch.tensor([0.4, -0.7], dtype=torch.float64)
    jacobian = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def function(input_params: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            input_params[0] + 2.0 * input_params[1],
            -input_params[0] + input_params[1],
        ))

    expected_ggn = jacobian.T @ (loss_hessian @ (jacobian @ vector))

    assert torch.allclose(vpx.dense_jacobian_anchor(function, params), jacobian)
    assert torch.allclose(
        vpx.ggnvp_dense_anchor(function, loss_hessian, params, vector),
        expected_ggn,
    )

    score_gradients = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    expected_fisher = score_gradients.T @ (score_gradients @ vector) / 3.0

    assert torch.allclose(
        vpx.fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )
    assert torch.allclose(
        vpx.empirical_fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )

    metric = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    inverse_product = vpx.dense_metric_inverse_multiply(metric, vector)

    assert torch.allclose(vpx.dense_metric_multiply(metric, vector), metric @ vector)
    assert torch.allclose(vpx.dense_metric_multiply(metric, inverse_product), vector)
    assert torch.allclose(
        vpx.dense_metric_inner(metric, vector, vector),
        vector @ (metric @ vector),
    )
    assert vpx.dense_metric_inverse_residual(metric, inverse_product, vector) < 1e-12


def test_measurement_timing_policy() -> None:
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 10.0

            return current

    samples, output, probe = measure_operation(
        operation,
        timing_policy=vpx.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=FakeClock(),
    )

    assert len(samples) == 5
    assert len(probe) == 1
    assert calls["count"] == 8
    assert torch.equal(output, torch.tensor(8.0))
    assert all(sample.device == "cpu" for sample in samples)


def test_measurement_reuses_long_probe_as_measured_sample() -> None:
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    samples, output, probe = measure_operation(
        operation,
        timing_policy=vpx.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 700.0)),
    )

    assert len(samples) == 1
    assert len(probe) == 1
    assert calls["count"] == 1
    assert torch.equal(output, torch.tensor(1.0))
    assert math.isclose(samples[0].elapsed_seconds, 700.0)


@pytest.mark.parametrize(
    ("error_cls", "message", "case", "error_type", "clock_end"),
    [
        (RuntimeError, "failed row", "runtime", "RuntimeError", 2.0),
        (
            torch.cuda.OutOfMemoryError,
            "cuda oom",
            "oom",
            "OutOfMemoryError",
            3.0,
        ),
    ],
)
def test_run_candidate_records_failures(
    error_cls: type[BaseException],
    message: str,
    case: str,
    error_type: str,
    clock_end: float,
) -> None:
    candidate = vpx.Candidate("family", "row", {})

    def operation() -> torch.Tensor:
        raise error_cls(message)

    record = run_candidate(
        candidate,
        {"case": case},
        operation,
        timing_policy=vpx.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, clock_end)),
    )

    assert record.status == "failed"
    assert record.error_type == error_type
    assert record.error == message
    assert record.reference_passed
    assert record.input_signature == {"case": case}
    assert record.timing_samples[0].elapsed_seconds == pytest.approx(clock_end)
    assert record.memory_samples[0].elapsed_seconds == pytest.approx(clock_end)


def test_run_candidate_records_compiled_selection_metadata() -> None:
    candidate = vpx.Candidate(
        "family",
        "compiled",
        {
            "compile.enabled": "true",
            "compile.cache_state": "cold_compile",
            "compile.compiled_autograd": "false",
            "compile.cuda_graphs": "false",
        },
    )
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    record = run_candidate(
        candidate,
        {"case": "compiled-metadata"},
        operation,
        timing_policy=vpx.TimingPolicy(
            short_seconds=10.0,
            short_warmups=0,
            short_measured_calls=2,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 5.0, 5.0, 7.0, 7.0, 9.0)),
    )

    assert record.status == "passed"
    assert record.selection_metadata == {
        "timing_source": "compiled_single_rank",
        "steady_elapsed_seconds": 2.0,
        "compile_time_seconds": 3.0,
        "recompile_count": 0,
        "compile_cache_state": "cold_compile",
        "compile.compiled_autograd": "false",
        "compile.cuda_graphs": "false",
    }


def test_candidate_rows_reject_missing_runtime_bindings() -> None:
    candidates = passed_candidates(
        "family",
        (
            ("baseline", {}),
            ("fused", {"fusion.loss": "fused_ce"}),
            (
                "packed",
                {"input.batch_layout": "packed_with_inverse_permutation"},
            ),
            ("lm-head", {"chunk.lm_head_weight_chunk_bytes": 1024}),
            ("layer-output", {"layout.output": "per_layer_flat"}),
            ("block-vector", {"layout.vector": "per_block_flat"}),
            ("layer-chunk", {"chunk.layer_block_size": 2}),
            ("mmap", {"memory.vector_residency": "mmap_cpu"}),
            ("intermediate", {"memory.intermediate_residency": "cpu_staged"}),
            ("manual", {"activation.recompute": "manual_recompute"}),
            ("teacher", {"teacher_outputs": "recomputed_with_equality_check"}),
            ("stateful-incomplete", {"call.path": "stateful_module"}),
            (
                "stateful-unbound",
                {
                    "call.path": "stateful_module",
                    "call.params": "module_params",
                    "call.buffers": "module_buffers",
                },
            ),
        ),
    )
    runtime = runtime_config(
        candidates,
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "fusion_rewriter": False,
            "batch_layout": False,
            "lm_head_chunker": False,
            "mmap_residency": False,
            "manual_recompute": False,
            "teacher_objective": False,
            "module": True,
            "module_call": None,
            "parameter_surface": {
                "layer_groups": (),
                "block_groups": (),
            },
        },
    )
    rows = {
        candidate.candidate_id: candidate
        for candidate in run_module._candidate_rows(runtime)
    }

    assert rows["baseline"].admission_status == "passed"
    assert rows["fused"].admission_error == (
        "fusion.loss requires a registered fused implementation"
    )
    assert rows["packed"].admission_error == (
        "declared input layout requires a batch_layout binding"
    )
    assert rows["lm-head"].admission_error == (
        "chunk.lm_head_weight_chunk_bytes requires an LM-head chunker binding"
    )
    assert rows["layer-output"].admission_error == (
        "layout.output=per_layer_flat requires declared layer_groups"
    )
    assert rows["block-vector"].admission_error == (
        "layout.vector=per_block_flat requires declared block_groups"
    )
    assert rows["layer-chunk"].admission_error == (
        "chunk.layer_block_size requires declared layer_groups"
    )
    assert rows["mmap"].admission_error == (
        "memory.vector_residency=mmap_cpu requires memory-mapped tensor metadata"
    )
    assert rows["intermediate"].admission_error == (
        "memory.intermediate_residency requires named intermediate boundaries"
    )
    assert rows["manual"].admission_status == "passed"
    assert rows["teacher"].admission_error == (
        "recomputed teacher outputs require a teacher objective"
    )
    assert rows["stateful-incomplete"].admission_error == (
        "call path settings are incomplete: ('call.params', 'call.buffers')"
    )
    assert rows["stateful-unbound"].admission_error == (
        "stateful_module requires a ModuleCallSpec binding"
    )

    for candidate_id, candidate in rows.items():
        if candidate_id not in {"baseline", "manual"}:
            assert candidate.admission_status == "failed"


def test_candidate_rows_reject_metric_representation_path_mismatches() -> None:
    def rows_for(
        operator_kind: str,
        representation_kind: str,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> Mapping[str, vpx.Candidate]:
        runtime = runtime_config(
            passed_candidates(operator_kind, rows),
            constant_operation_factory,
            passing_reference_check,
            materialize_candidate,
            None,
            {
                "runtime": "standard",
                "operator": {
                    "kind": operator_kind,
                    "semantics": {
                        "representation": {"kind": representation_kind},
                    },
                },
            },
        )

        return {
            candidate.candidate_id: candidate
            for candidate in run_module._candidate_rows(runtime)
        }

    dense_metric_rows = rows_for(
        "metric",
        "dense_matrix",
        (
            ("dense-ok", {"metric.multiply_path": "dense_matmul"}),
            ("dense-factorized", {"metric.multiply_path": "factorized_multiply"}),
            ("dense-streaming", {"metric.multiply_path": "streaming_multiply"}),
        ),
    )
    matrix_free_rows = rows_for(
        "metric",
        "matrix_free",
        (
            ("streaming-ok", {"metric.multiply_path": "streaming_multiply"}),
            ("matrix-free-dense", {"metric.multiply_path": "dense_matmul"}),
        ),
    )
    diagonal_inverse_rows = rows_for(
        "inverse_metric",
        "diagonal_tree",
        (
            ("factorized-ok", {"inverse_metric.solve_path": "factorized_solve"}),
            ("dense-inverse", {"inverse_metric.solve_path": "dense_solve"}),
        ),
    )
    dense_sqrt_rows = rows_for(
        "sqrt_metric",
        "dense_matrix",
        (
            ("cholesky-ok", {"sqrt_metric.factor_path": "cholesky_factor"}),
            ("lanczos", {"sqrt_metric.factor_path": "matrix_free_lanczos"}),
            (
                "closed-form",
                {"sqrt_metric.factor_path": "closed_form_factor_square_root"},
            ),
        ),
    )

    assert dense_metric_rows["dense-ok"].admission_status == "passed"
    assert dense_metric_rows["dense-factorized"].admission_error == (
        "factorized metric path is not lowered for representation: dense_matrix"
    )
    assert dense_metric_rows["dense-streaming"].admission_error == (
        "metric streaming path is not lowered for representation: dense_matrix"
    )
    assert matrix_free_rows["streaming-ok"].admission_status == "passed"
    assert matrix_free_rows["matrix-free-dense"].admission_error == (
        "matrix_free metric requires streaming_multiply"
    )
    assert diagonal_inverse_rows["factorized-ok"].admission_status == "passed"
    assert diagonal_inverse_rows["dense-inverse"].admission_error == (
        "metric representation kind is not supported by path: diagonal_tree"
    )
    assert dense_sqrt_rows["cholesky-ok"].admission_status == "passed"
    assert dense_sqrt_rows["lanczos"].admission_error == (
        "metric representation kind is not supported by path: dense_matrix"
    )
    assert dense_sqrt_rows["closed-form"].admission_error == (
        "metric representation kind is not supported by path: dense_matrix"
    )


def test_candidate_rows_admit_declared_parameter_groups() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "family",
                "grouped",
                {
                    "layout.output": "per_layer_flat",
                    "layout.vector": "per_block_flat",
                    "chunk.layer_block_size": 1,
                },
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "parameter_surface": {
                "layer_groups": (("w",),),
                "block_groups": (("w",),),
            },
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "passed"


def test_candidate_rows_reject_fusion_without_module() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "family",
                "fused",
                {"fusion.loss": "fused_ce"},
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "fusion_rewriter": True,
            "module": False,
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == "fused rows require a module"


def test_candidate_rows_reject_fused_loss_without_loss_identity() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "family",
                "fused",
                {"fusion.loss": "fused_ce"},
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "fusion_rewriter": True,
            "module": True,
            "operator": ops.hvp("family", "loss", aggregation="sum").signature(),
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == (
        "fusion.loss=fused_ce requires typed softmax_cross_entropy loss identity"
    )


def test_candidate_rows_reject_sampled_fisher_exact_check_without_bound() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "sampled",
                "exact-check",
                {"sampled_fisher.exact_fisher_check": ("enabled_with_sampling_bound")},
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "operator": ops.sampled_fisher_vp(
                "sampled",
                "scores",
                aggregation="mean_per_example",
                distribution="explicit_score_gradients",
                label_policy="sampled_labels",
                sample_count=2,
                sample_source="fixed_seed_and_count",
                sampling_bound={"kind": "disabled"},
                score_reduction="none",
                denominator="num_examples",
            ).signature(),
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == (
        "sampled_fisher exact-Fisher check requires declared sampling_bound"
    )


@pytest.mark.parametrize(
    "sampling_bound",
    [
        {
            "kind": "matrix_bernstein",
            "failure_probability": 0.5,
            "norm_floor": 1e-12,
        },
        {
            "kind": "hutchinson_relative_variance",
            "failure_probability": 0.5,
            "norm_floor": 1e-12,
        },
    ],
)
def test_candidate_rows_accept_sampled_fisher_named_bounds(
    sampling_bound: Mapping[str, object],
) -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "sampled",
                "exact-check",
                {"sampled_fisher.exact_fisher_check": "enabled_with_sampling_bound"},
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "operator": ops.sampled_fisher_vp(
                "sampled",
                "scores",
                aggregation="mean_per_example",
                distribution="explicit_score_gradients",
                label_policy="sampled_labels",
                sample_count=2,
                sample_source="fixed_seed_and_count",
                sampling_bound=sampling_bound,
                score_reduction="none",
                denominator="num_examples",
            ).signature(),
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "passed"


def test_candidate_rows_reject_stateful_module_without_module() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "family",
                "stateful",
                {
                    "call.path": "stateful_module",
                    "call.params": "module_params",
                    "call.buffers": "module_buffers",
                },
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "module": False,
            "module_call": {"positional_batch_keys": ("scale",)},
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "failed"
    assert rows[0].admission_error == "call.path=stateful_module requires a module"


def test_candidate_rows_admit_builtin_intermediate_residency_points() -> None:
    runtime = runtime_config(
        (
            vpx.Candidate(
                "family",
                "ggn-intermediate",
                {"memory.intermediate_residency": "cpu_staged"},
                admission_status="passed",
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "operator": {"kind": "ggnvp"},
        },
    )
    rows = tuple(run_module._candidate_rows(runtime))

    assert rows[0].admission_status == "passed"


def test_run_candidate_records_full_size_check_metadata() -> None:
    candidate = vpx.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
    )

    def operation() -> torch.Tensor:
        return torch.tensor([1.0])

    def full_size_check(
        output: vpx.TensorTree,
        samples: tuple[vpx.Measurement, ...],
    ) -> Mapping[str, object]:
        assert isinstance(output, torch.Tensor)
        assert len(samples) == 1
        torch.testing.assert_close(output, torch.tensor([1.0]))

        return {
            "full_size_agreement_passed": True,
            "full_size_agreement_name": "tests.full_size_gate",
        }

    record = run_candidate(
        candidate,
        {"case": "full-size-check"},
        operation,
        timing_policy=vpx.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
        full_size_check=full_size_check,
    )

    assert record.status == "passed"
    assert record.selection_metadata["full_size_agreement_passed"] is True
    assert record.selection_metadata["full_size_agreement_name"] == (
        "tests.full_size_gate"
    )


def test_run_candidate_records_measured_recompile_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = vpx.Candidate(
        "family",
        "compiled",
        {
            "compile.enabled": "true",
            "compile.cache_state": "cold_compile",
            "compile.compiled_autograd": "false",
            "compile.cuda_graphs": "false",
        },
    )
    counters = {"stats": {"unique_graphs": 10}}
    calls = {"count": 0}

    fake_dynamo_utils = types.SimpleNamespace(counters=counters)

    def fake_import_module(name: str) -> object:
        assert name == "torch._dynamo.utils"

        return fake_dynamo_utils

    def operation() -> torch.Tensor:
        calls["count"] += 1

        if calls["count"] == 1:
            counters["stats"]["unique_graphs"] += 3

        return torch.tensor(float(calls["count"]))

    monkeypatch.setattr(measure_module.importlib, "import_module", fake_import_module)
    record = run_candidate(
        candidate,
        {"case": "compiled-recompile-count"},
        operation,
        timing_policy=vpx.TimingPolicy(
            short_seconds=10.0,
            short_warmups=0,
            short_measured_calls=2,
        ),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 5.0, 5.0, 7.0, 7.0, 9.0)),
    )

    assert record.selection_metadata["recompile_count"] == 2


def test_measurement_cleans_memory_backend_after_runtime_failure() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.cleanup_calls = 0

        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"backend_id": "tests.recording_memory"}

        def prepare(self) -> None:
            pass

        @staticmethod
        def sample() -> tuple[Measurement, ...]:
            return (
                Measurement(
                    elapsed_seconds=0.0,
                    peak_allocated_mib=0.0,
                    peak_reserved_mib=0.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                ),
            )

        def synchronize(self) -> None:
            pass

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    backend = RecordingBackend()

    def operation() -> torch.Tensor:
        message = "failed measurement"
        raise RuntimeError(message)

    with pytest.raises(vp.MeasurementError) as error_info:
        measure_once(operation, memory_backend=backend)

    error = error_info.value

    assert isinstance(error, measure_module.OperationMeasurementError)
    assert error.error_type == "RuntimeError"
    assert error.samples[0].device == "cpu"
    assert backend.cleanup_calls == 2


def _record(
    candidate: vpx.Candidate,
    *,
    elapsed: tuple[float, ...],
    reserved: tuple[float, ...],
    input_signature: Mapping[str, object],
    status: str = "passed",
    reference_passed: bool = True,
    selection_metadata: Mapping[str, object] | None = None,
) -> FullSizeRecord:
    samples = tuple(
        Measurement(
            elapsed_seconds=time,
            peak_allocated_mib=memory,
            peak_reserved_mib=memory,
            post_allocated_mib=0.0,
            post_reserved_mib=0.0,
        )
        for time, memory in zip(elapsed, reserved, strict=True)
    )

    record = FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status=status,
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=samples,
        memory_samples=samples,
        selection_metadata={}
        if selection_metadata is None
        else dict(selection_metadata),
        dependency_identities=dict(candidate.dependency_identities),
        reference_passed=reference_passed,
    )

    return record


def _with_rank_memory_samples(
    record: FullSizeRecord,
    reserved: tuple[float, ...],
) -> FullSizeRecord:
    samples = tuple(
        Measurement(
            elapsed_seconds=record.timing_samples[0].elapsed_seconds,
            peak_allocated_mib=memory,
            peak_reserved_mib=memory,
            post_allocated_mib=0.0,
            post_reserved_mib=0.0,
            rank=rank,
            device=f"cuda:{rank}",
        )
        for rank, memory in enumerate(reserved)
    )

    return dataclasses.replace(record, memory_samples=samples)


def _check_record(
    candidate: vpx.Candidate,
    *,
    input_signature: Mapping[str, object],
) -> vpx.CheckRecord:
    record = vpx.CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name="tree_close",
        status="passed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds={"max_abs_diff": 1e-6},
        measurements={"max_abs_diff": 0.0},
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        dependency_identities=dict(candidate.dependency_identities),
    )

    return record


def _identity_kwargs(
    families: tuple[str, ...] = ("family",),
) -> dict[str, Any]:
    return {
        "target_identity": {"target": "test", "environment": {}},
        "runtime_identities": {
            family: {"runtime": f"test.{family}"} for family in families
        },
        "adapter_identities": {
            family: {"adapter_id": "tests", "adapter_version": "1"}
            for family in families
        },
    }


def test_plan_materialize_validates_selected_dependency_identities() -> None:
    input_signature = {"case": "plan-materialize-dependencies"}
    dependency = vpx.Candidate("dependency", "selected", {}, admission_status="passed")
    dependent = vpx.Candidate(
        "dependent",
        "selected",
        {},
        dependency_identities={"dependency": {"stale": "identity"}},
        admission_status="passed",
    )
    dependency_record = _record(
        dependency,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    dependent_record = _record(
        dependent,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vpx.Plan(
        selected={
            "dependency": dependency,
            "dependent": dependent,
        },
        records={
            "dependency": dependency_record,
            "dependent": dependent_record,
        },
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        materializers={
            "dependency": materialize_candidate,
            "dependent": materialize_candidate,
        },
        dependencies_by_family={
            "dependency": (),
            "dependent": ("dependency",),
        },
    )

    with pytest.raises(vp.MaterializationError, match="dependency identity"):
        plan.materialize("dependent")


def test_plan_materialize_accepts_public_name_selector() -> None:
    input_signature = {"case": "plan-materialize-name"}
    candidate = vpx.Candidate(
        "family",
        "selected",
        {"scale": 3.0},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        materializers={"family": materialize_candidate},
        dependencies_by_family={"family": ()},
    )

    selected_from_plan = plan.materialize(name="family")
    selected_from_public_helper = vp.materialize(plan, name="family")

    assert torch.equal(selected_from_plan(), torch.tensor([3.0]))
    assert torch.equal(selected_from_public_helper(), torch.tensor([3.0]))

    with pytest.raises(vp.MaterializationError, match="selectors differ"):
        plan.materialize("family", name="other")

    public_materialize = vars(vp)["materialize"]

    with pytest.raises(TypeError, match="unexpected keyword argument 'family'"):
        public_materialize(plan, family="family")


def test_validate_plan_materializes_in_validation_order() -> None:
    input_signature = {"case": "validation-materialization-order"}
    calls = []

    def first_materializer_callback(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
    ) -> str:
        assert candidate.family == record.family
        calls.append("materialize:first")

        return "first"

    def second_materializer_callback(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
    ) -> str:
        assert candidate.family == record.family
        calls.append("materialize:second")

        return "second"

    first_materializer = vpx.CallableMaterializer(
        "tests.first_materializer",
        "1",
        {},
        {"callback": "tests.first_materializer_callback"},
        first_materializer_callback,
    )
    second_materializer = vpx.CallableMaterializer(
        "tests.second_materializer",
        "1",
        {},
        {"callback": "tests.second_materializer_callback"},
        second_materializer_callback,
    )
    first = vpx.Candidate("first", "row", {}, admission_status="passed")
    first_record = _record(
        first,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    first_identity = {
        "family": "first",
        "candidate_id": "row",
        "candidate_settings": dict(first.settings),
        "full_size_row": first_record.row_key(),
        "materializer_identity": dict(first_materializer.identity()),
    }
    second = vpx.Candidate(
        "second",
        "row",
        {},
        dependency_identities={"first": first_identity},
        admission_status="passed",
    )
    second_record = _record(
        second,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vpx.Plan(
        selected={"first": first, "second": second},
        records={"first": first_record, "second": second_record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        materializers={
            "first": first_materializer,
            "second": second_materializer,
        },
        validation_order=("first", "second"),
        dependencies_by_family={"first": (), "second": ("first",)},
    )

    def first_validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == record.family
        assert context.selected == "first"
        calls.append("validate:first")
        message = "stop before downstream materialization"
        raise RuntimeError(message)

    def second_validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == record.family
        assert context.selected == "second"
        calls.append("validate:second")

        return reference_passed()

    with pytest.raises(RuntimeError, match="downstream materialization"):
        vp.validate_plan(
            plan,
            {"first": first_validator, "second": second_validator},
        )

    assert calls == ["materialize:first", "validate:first"]


def _current_record(record: FullSizeRecord) -> FullSizeRecord:
    return record


def test_within_family_selection() -> None:
    signature = {"case": "current"}
    policy = vpx.SelectionPolicy()
    fast = vpx.Candidate("family", "fast", {})
    near = vpx.Candidate("family", "near", {})
    slow = vpx.Candidate("family", "slow", {})

    selected, _ = select_family(
        (
            (
                fast,
                _record(
                    fast,
                    elapsed=(10.0, 10.0),
                    reserved=(9.0, 9.0),
                    input_signature=signature,
                ),
            ),
            (
                near,
                _record(
                    near,
                    elapsed=(10.4, 10.4),
                    reserved=(1.0, 1.0),
                    input_signature=signature,
                ),
            ),
            (
                slow,
                _record(
                    slow,
                    elapsed=(11.0, 11.0),
                    reserved=(0.1, 0.1),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected == near


@pytest.mark.parametrize(
    "settings",
    [
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
        },
        {"attention.frontend": "transformers_flex_attention"},
        {"attention.frontend": "unknown_eager"},
        {
            "attention.frontend": "packed_exact",
            "attention.partition": "packed_tokens",
        },
        {
            "attention.frontend": "blockwise_exact",
            "attention.partition": "blockwise_queries",
        },
        {"compile.cuda_graphs": "true"},
        {"compile.mode": "max-autotune"},
        {"fusion.loss": "fused_ce"},
        {"tp.loss_parallel": "true"},
        {"context_parallel.enabled": "true"},
    ],
)
def test_selection_requires_full_size_agreement(settings: Mapping[str, object]) -> None:
    signature = {"case": "full-size-gate", "settings": dict(settings)}
    policy = vpx.SelectionPolicy()
    gated = vpx.Candidate(
        "family",
        "gated",
        settings,
    )
    fallback = vpx.Candidate("family", "fallback", {})

    selected, _ = select_family(
        (
            (
                gated,
                _record(
                    gated,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
            (
                fallback,
                _record(
                    fallback,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected == fallback

    selected_with_gate, _ = select_family(
        (
            (
                gated,
                _record(
                    gated,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                    selection_metadata={"full_size_agreement_passed": True},
                ),
            ),
            (
                fallback,
                _record(
                    fallback,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected_with_gate == gated


@pytest.mark.parametrize(
    "settings",
    [
        {"attention.frontend": "pytorch_sdpa_direct"},
        {"attention.frontend": "patched_eager"},
        {"attention.frontend": "transformers_eager"},
        {"attention.frontend": "transformers_sdpa"},
        {"attention.sdpa_kernel": "math"},
        {
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("math",),
        },
        {"distributed.strategy": "single_gpu"},
    ],
)
def test_selection_allows_baseline_rows_without_full_size_agreement(
    settings: Mapping[str, object],
) -> None:
    signature = {"case": "ungated-selection", "settings": dict(settings)}
    policy = vpx.SelectionPolicy()
    fast = vpx.Candidate("family", "fast", settings)
    slow = vpx.Candidate("family", "slow", {})

    selected, _ = select_family(
        (
            (
                fast,
                _record(
                    fast,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
            (
                slow,
                _record(
                    slow,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected == fast


def test_selection_scores_compiled_rows_by_call_horizon() -> None:
    signature = {"case": "compiled-selection"}
    eager = vpx.Candidate("family", "eager", {})
    compiled = vpx.Candidate(
        "family",
        "compiled",
        {"compile.enabled": "true"},
    )
    eager_record = _record(
        eager,
        elapsed=(4.0,),
        reserved=(1.0,),
        input_signature=signature,
    )
    compiled_record = _record(
        compiled,
        elapsed=(2.0,),
        reserved=(1.0,),
        input_signature=signature,
        selection_metadata={
            "steady_elapsed_seconds": 2.0,
            "compile_time_seconds": 30.0,
            "recompile_count": 0,
        },
    )
    short_horizon = vpx.SelectionPolicy(compile_call_horizon=10)
    long_horizon = vpx.SelectionPolicy(compile_call_horizon=30)

    short_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=short_horizon,
    )
    long_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=long_horizon,
    )

    assert short_selected == eager
    assert long_selected == compiled


def test_selection_scores_distributed_rows_by_global_elapsed_seconds() -> None:
    signature = {"case": "distributed-selection"}
    local = vpx.Candidate("family", "local", {})
    distributed = vpx.Candidate(
        "family",
        "distributed",
        {"distributed.strategy": "fsdp2"},
    )
    selected, _ = select_family(
        (
            (
                local,
                _record(
                    local,
                    elapsed=(2.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                ),
            ),
            (
                distributed,
                _record(
                    distributed,
                    elapsed=(1.0,),
                    reserved=(1.0,),
                    input_signature=signature,
                    selection_metadata={"global_elapsed_seconds": 3.0},
                ),
            ),
        ),
        input_signature=signature,
        policy=vpx.SelectionPolicy(),
    )

    assert selected == local

    with pytest.raises(TypeError, match="global_elapsed_seconds"):
        select_family(
            (
                (
                    distributed,
                    _record(
                        distributed,
                        elapsed=(1.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            ),
            input_signature=signature,
            policy=vpx.SelectionPolicy(),
        )


def test_selection_scores_compiled_distributed_rows_by_global_compile_fields() -> None:
    signature = {"case": "compiled-distributed-selection"}
    eager = vpx.Candidate("family", "eager", {})
    compiled = vpx.Candidate(
        "family",
        "compiled",
        {"compile.enabled": "true", "distributed.strategy": "fsdp2"},
    )
    eager_record = _record(
        eager,
        elapsed=(4.0,),
        reserved=(1.0,),
        input_signature=signature,
    )
    compiled_record = _record(
        compiled,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=signature,
        selection_metadata={
            "global_steady_elapsed_seconds": 1.0,
            "global_compile_time_seconds": 90.0,
            "recompile_count": 0,
        },
    )
    short_horizon = vpx.SelectionPolicy(compile_call_horizon=10)
    long_horizon = vpx.SelectionPolicy(compile_call_horizon=90)

    short_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=short_horizon,
    )
    long_selected, _ = select_family(
        ((eager, eager_record), (compiled, compiled_record)),
        input_signature=signature,
        policy=long_horizon,
    )

    assert short_selected == eager
    assert long_selected == compiled


def test_selection_tie_breaks_with_declared_rank_memory_reduction() -> None:
    signature = {"case": "distributed-memory-selection"}
    first = vpx.Candidate("family", "first", {"distributed.strategy": "fsdp2"})
    second = vpx.Candidate("family", "second", {"distributed.strategy": "fsdp2"})
    first_record = _with_rank_memory_samples(
        _record(
            first,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
            selection_metadata={"global_elapsed_seconds": 1.0},
        ),
        (60.0, 1.0),
    )
    second_record = _with_rank_memory_samples(
        _record(
            second,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
            selection_metadata={"global_elapsed_seconds": 1.0},
        ),
        (40.0, 40.0),
    )
    max_selected, _ = select_family(
        ((first, first_record), (second, second_record)),
        input_signature=signature,
        policy=vpx.SelectionPolicy(rank_memory_reduction="max_peak_reserved"),
    )
    sum_selected, _ = select_family(
        ((first, first_record), (second, second_record)),
        input_signature=signature,
        policy=vpx.SelectionPolicy(rank_memory_reduction="sum_peak_reserved"),
    )

    assert max_selected == second
    assert sum_selected == first


def test_cohort_selection_sums_compiled_row_scores() -> None:
    signature = {"case": "compiled-cohort"}
    eager_a = vpx.Candidate("a", "eager-a", {})
    eager_b = vpx.Candidate("b", "eager-b", {})
    compiled_a = vpx.Candidate("a", "compiled-a", {"compile.enabled": "true"})
    compiled_b = vpx.Candidate("b", "compiled-b", {"compile.enabled": "true"})
    policy = vpx.SelectionPolicy(compile_call_horizon=30)
    cohort = select_cohort(
        (
            {
                "a": (
                    eager_a,
                    _record(
                        eager_a,
                        elapsed=(4.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    eager_b,
                    _record(
                        eager_b,
                        elapsed=(4.0,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            },
            {
                "a": (
                    compiled_a,
                    _record(
                        compiled_a,
                        elapsed=(2.0,),
                        reserved=(2.0,),
                        input_signature=signature,
                        selection_metadata={
                            "steady_elapsed_seconds": 2.0,
                            "compile_time_seconds": 30.0,
                            "recompile_count": 0,
                        },
                    ),
                ),
                "b": (
                    compiled_b,
                    _record(
                        compiled_b,
                        elapsed=(2.0,),
                        reserved=(2.0,),
                        input_signature=signature,
                        selection_metadata={
                            "steady_elapsed_seconds": 2.0,
                            "compile_time_seconds": 30.0,
                            "recompile_count": 0,
                        },
                    ),
                ),
            },
        ),
        families=("a", "b"),
        policy=policy,
    )

    assert cohort["a"][0] == compiled_a
    assert cohort["b"][0] == compiled_b


@pytest.mark.parametrize("value", [0.0, 0.99, float("nan"), True])
def test_selection_policy_rejects_invalid_near_fastest_multiplier(
    value: float | bool,
) -> None:
    with pytest.raises(RuntimeError, match="near_fastest_multiplier"):
        vpx.SelectionPolicy(near_fastest_multiplier=value)


@pytest.mark.parametrize("value", [0, True, unchecked_timing_policy_value(1.5)])
def test_selection_policy_rejects_invalid_compile_call_horizon(value: Any) -> None:
    with pytest.raises(RuntimeError, match="compile_call_horizon"):
        vpx.SelectionPolicy(compile_call_horizon=value)


def test_selection_rejects_unsupported_policy_fields() -> None:
    candidate = vpx.Candidate("family", "row", {})
    policies = (
        vpx.SelectionPolicy(speed_statistic="mean_elapsed_seconds"),
        vpx.SelectionPolicy(compiled_speed_statistic="steady_elapsed_seconds"),
        vpx.SelectionPolicy(distributed_speed_statistic="rank_zero_elapsed_seconds"),
        vpx.SelectionPolicy(rank_memory_reduction="rank_zero_peak_reserved"),
        vpx.SelectionPolicy(cohort_speed_statistic="sum_selection_score_seconds"),
        vpx.SelectionPolicy(accepted_status="passed_only"),
    )

    for policy in policies:
        with pytest.raises(RuntimeError):
            select_family(
                (
                    (
                        candidate,
                        _record(
                            candidate,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature={},
                        ),
                    ),
                ),
                input_signature={},
                policy=policy,
            )


def test_cohort_constraint_uses_spec_selection_aggregation_token() -> None:
    constraint = vpx.CohortConstraint(
        name="dtype",
        settings_keys=("dtype.model_compute",),
        assignments=({"dtype.model_compute": "fp32"},),
    )

    assert constraint.selection_aggregation == "sum_median_elapsed_seconds"

    with pytest.raises(RuntimeError):
        vpx.CohortConstraint(
            name="old-token",
            settings_keys=("dtype.model_compute",),
            assignments=({"dtype.model_compute": "fp32"},),
            selection_aggregation="sum_selection_score_seconds",
        )


def test_selection_rejects_invalid_rows() -> None:
    signature = {"case": "current"}
    stale = vpx.Candidate("family", "stale", {})
    failed = vpx.Candidate("family", "failed", {})

    with pytest.raises(vp.NoPassedCandidateError, match="accepted rows"):
        select_family(
            (
                (
                    stale,
                    _record(
                        stale,
                        elapsed=(1.0,),
                        reserved=(1.0,),
                        input_signature={"case": "old"},
                    ),
                ),
                (
                    failed,
                    _record(
                        failed,
                        elapsed=(),
                        reserved=(),
                        input_signature=signature,
                        status="failed",
                    ),
                ),
            ),
            input_signature=signature,
            policy=vpx.SelectionPolicy(),
        )


def test_selection_rejects_record_from_different_candidate_settings() -> None:
    signature = {"case": "current"}
    candidate = vpx.Candidate("family", "row", {})
    mismatched = dataclasses.replace(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
        ),
        candidate_settings={"axis": "different"},
    )

    with pytest.raises(vp.NoPassedCandidateError):
        select_family(
            ((candidate, mismatched),),
            input_signature=signature,
            policy=vpx.SelectionPolicy(),
        )


def test_selection_uses_json_normalized_signatures() -> None:
    candidate = vpx.Candidate("family", "row", {})
    tuple_signature = {"shape": (1, 2)}
    list_signature = {"shape": [1, 2]}
    record = _current_record(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=tuple_signature,
        )
    )
    selected, _ = select_family(
        ((candidate, record),),
        input_signature=list_signature,
        policy=vpx.SelectionPolicy(),
    )

    assert selected == candidate


def test_memory_stability() -> None:
    candidate = vpx.Candidate("family", "row", {})
    stable = _record(
        candidate,
        elapsed=(1.0, 1.0),
        reserved=(1.0, 1.0),
        input_signature={},
    )
    unstable = FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature={},
        candidate_settings={},
        generator_id="default",
        generator_version="0.0.1",
        timing_samples=stable.timing_samples,
        memory_samples=(
            Measurement(1.0, 1.0, 1.0, 1.0, 1.0),
            Measurement(1.0, 1.0, 1.0, 1.0, 2.0),
        ),
    )

    assert memory_stable(stable)
    assert not memory_stable(unstable)

    mixed_device_unstable = dataclasses.replace(
        stable,
        memory_samples=(
            Measurement(1.0, 1.0, 1.0, 100.0, 100.0, rank=0, device="cuda:0"),
            Measurement(1.0, 1.0, 1.0, 1.0, 1.0, rank=0, device="cuda:1"),
            Measurement(1.0, 1.0, 1.0, 2.0, 2.0, rank=0, device="cuda:1"),
        ),
    )

    assert not memory_stable(mixed_device_unstable)


def test_cohort_selection_prefers_lower_memory_near_fastest() -> None:
    signature = {}
    policy = vpx.SelectionPolicy()
    first_a = vpx.Candidate("a", "first-a", {"backend": "first"})
    first_b = vpx.Candidate("b", "first-b", {"backend": "first"})
    second_a = vpx.Candidate("a", "second-a", {"backend": "second"})
    second_b = vpx.Candidate("b", "second-b", {"backend": "second"})
    cohort = select_cohort(
        (
            {
                "a": (
                    first_a,
                    _record(
                        first_a,
                        elapsed=(10.0,),
                        reserved=(10.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    first_b,
                    _record(
                        first_b,
                        elapsed=(10.0,),
                        reserved=(10.0,),
                        input_signature=signature,
                    ),
                ),
            },
            {
                "a": (
                    second_a,
                    _record(
                        second_a,
                        elapsed=(10.4,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    second_b,
                    _record(
                        second_b,
                        elapsed=(10.4,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            },
        ),
        families=("a", "b"),
        policy=policy,
    )

    assert cohort["a"][0] == second_a


def test_cohort_selection_rejects_incomplete_cohort() -> None:
    signature = {}
    first_a = vpx.Candidate("a", "first-a", {"backend": "first"})
    second_a = vpx.Candidate("a", "second-a", {"backend": "second"})
    second_b = vpx.Candidate("b", "second-b", {"backend": "second"})

    with pytest.raises(vp.NoPassedCandidateError):
        select_cohort(
            (
                {
                    "a": (
                        first_a,
                        _record(
                            first_a,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    )
                },
                {
                    "a": (
                        second_a,
                        _record(
                            second_a,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    ),
                    "b": (
                        second_b,
                        _record(
                            second_b,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    ),
                },
            ),
            families=("a", "b", "c"),
            policy=vpx.SelectionPolicy(),
        )


def test_record_current_rejects_stale_generator_version() -> None:
    record = {
        "record_type": "full_size",
        "schema_version": 1,
        "package_version": "0.0.1",
        "input_signature": {},
        "candidate_settings": {},
        "status": "passed",
        "generator_id": "gen",
        "generator_version": "1",
    }

    assert not record_current(
        record,
        record_type="full_size",
        family="family",
        candidate_id="row",
        input_signature={},
        candidate_settings={},
        generator_id="gen",
        generator_version="2",
    )


def test_record_current_rejects_stale_candidate_and_full_size_status() -> None:
    input_signature = _input_signature("status-current")
    candidate = vpx.Candidate(
        "family",
        "row",
        {"dtype": "fp32"},
        admission_status="passed",
    )
    candidate_row = vpx.candidate_record_to_json(candidate, input_signature)

    assert record_current(
        candidate_row,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert not record_current(
        candidate_row,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="failed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )

    full_size_row = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    full_size_json = vpx.full_size_record_to_json(full_size_row)

    assert full_size_row.row_key()["status"] == "passed"
    assert record_current(
        full_size_json,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert not record_current(
        full_size_json,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="failed",
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )


def test_reference_row_key_includes_row_and_check_identity() -> None:
    candidate_a = vpx.Candidate("family", "row-a", {"axis": "same"})
    candidate_b = vpx.Candidate("family", "row-b", {"axis": "same"})
    first = _check_record(candidate_a, input_signature={})
    second = _check_record(candidate_b, input_signature={})
    third = dataclasses.replace(first, name="second")
    fourth = dataclasses.replace(first, status="failed")
    fifth = dataclasses.replace(first, thresholds={"max_abs_diff": 1e-3})

    assert first.row_key() != second.row_key()
    assert first.row_key() != third.row_key()
    assert first.row_key() != fourth.row_key()
    assert first.row_key() != fifth.row_key()


def test_check_record_current_rejects_stale_status_and_thresholds() -> None:
    record = _check_record(vpx.Candidate("family", "row", {}), input_signature={})

    assert not vpx.check_record_current(dataclasses.replace(record, status="failed"))
    assert not vpx.check_record_current(
        dataclasses.replace(record, thresholds={"max_abs_diff": 1e-3})
    )


def test_admission_helpers() -> None:
    vpx.admit_functional_call({
        "parameter_keys": ("weight",),
        "buffer_keys": ("running",),
        "tie_weights": True,
        "strict": False,
        "parametrization_policy": "active",
        "mutates_state": False,
        "mutated_parameter_keys": (),
        "mutated_buffer_keys": (),
        "module_mode": "eval",
    })

    with pytest.raises(vp.AdmissionError):
        vpx.admit_torch_func({
            "contains_autograd_call": True,
            "contains_backward_call": False,
            "uses_out_variant": False,
            "uses_data_dependent_control_flow": False,
            "uses_item": False,
            "has_dynamic_shape_output": False,
            "vectorization.randomness": "error",
            "requires_forward_ad": False,
            "forward_ad_supported": False,
        })

    with pytest.raises(vp.AdmissionError):
        vpx.admit_checkpoint({
            "checkpoint.use_reentrant": "true",
            "checkpoint.preserve_rng_state": "true",
            "checkpoint.determinism_check": "default",
            "checkpoint.context_fn": "none",
            "checkpoint.early_stop": "true",
            "checkpoint.moves_to_new_device": "false",
            "checkpoint.uses_global_state": "false",
        })


def checkpoint_fields() -> dict[str, object]:
    return {
        "activation.offload": "none",
        "checkpoint.use_reentrant": "false",
        "checkpoint.preserve_rng_state": "true",
        "checkpoint.determinism_check": "default",
        "checkpoint.context_fn": "none",
        "checkpoint.early_stop": "true",
        "checkpoint.moves_to_new_device": "false",
        "checkpoint.uses_global_state": "false",
    }


def test_checkpoint_operation_preserves_rng_state() -> None:
    values = []
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(value)
        values.append(noise.detach().clone())

        return (value * noise).sum()

    torch.manual_seed(17)
    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert len(values) == 2
    assert torch.equal(values[0], values[1])


def test_checkpoint_operation_can_disable_rng_preservation() -> None:
    values = []
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.preserve_rng_state": "false",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(value)
        values.append(noise.detach().clone())

        return (value * noise).sum()

    torch.manual_seed(17)
    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert len(values) == 2
    assert not torch.equal(values[0], values[1])


@pytest.mark.parametrize(
    "recompute",
    [
        "checkpoint_non_reentrant_by_layer",
        "checkpoint_selective",
    ],
)
def test_checkpoint_operation_uses_declared_context_pair(recompute: str) -> None:
    events = []
    vector = torch.tensor([1.0], requires_grad=True)

    def context_fn() -> tuple[object, object]:
        return contextlib.nullcontext(), contextlib.nullcontext()

    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": recompute,
            **checkpoint_fields(),
            "checkpoint.context_fn": "declared_context_pair",
            "checkpoint.context_fn_callable": "default",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        events.append("called")

        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
        checkpoint_contexts={"default": context_fn},
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert events == ["called", "called"]


def test_checkpoint_operation_rejects_declared_context_without_context_id() -> None:
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.context_fn": "declared_context_pair",
        },
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square()

    with pytest.raises(vp.AdmissionError, match="requires context id"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="activation.recompute",
        )()


def test_checkpoint_operation_rejects_unadmitted_fields_before_execution() -> None:
    calls = []
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "checkpoint_non_reentrant_by_layer",
            **checkpoint_fields(),
            "checkpoint.use_reentrant": "true",
        },
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.detach().clone())

        return value.square()

    with pytest.raises(vp.AdmissionError):
        vpx.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="activation.recompute",
        )()

    assert calls == []


def test_checkpoint_operation_runs_cpu_saved_tensor_hooks() -> None:
    vector = torch.tensor([2.0], requires_grad=True)
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "saved_tensor_hooks_cpu",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert torch.equal(vector.grad, torch.tensor([4.0]))


def test_checkpoint_operation_runs_custom_saved_tensor_hooks() -> None:
    events = []
    vector = torch.tensor([2.0], requires_grad=True)

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("pack", tensor.detach().clone()))

        return tensor.detach().clone()

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        events.append(("unpack", tensor.detach().clone()))

        return tensor

    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "custom_saved_tensor_hooks",
            "activation.pack_hook": "pack",
            "activation.unpack_hook": "unpack",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum()

    output = vpx.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="activation.recompute",
        activation_pack_hooks={"pack": pack_hook},
        activation_unpack_hooks={"unpack": unpack_hook},
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert tuple(event for event, _ in events) == ("pack", "unpack")
    assert vector.grad is not None
    assert torch.equal(vector.grad, torch.tensor([4.0]))


def test_checkpoint_operation_rejects_missing_activation_offload() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"activation.recompute": "none"},
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(vp.AdmissionError, match=r"activation[.]offload"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (torch.tensor([1.0]),),
            policy_key="activation.recompute",
        )


def test_checkpoint_operation_rejects_custom_offload_without_hooks() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {
            "activation.recompute": "none",
            "activation.offload": "custom_saved_tensor_hooks",
        },
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(vp.AdmissionError, match="pack and unpack"):
        vpx.checkpoint_operation(
            candidate,
            function,
            (torch.tensor([1.0]),),
            policy_key="activation.recompute",
        )()


def test_adapter_runtime_executes_checkpoint_without_standard_runtime_support(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    settings = {
        "operator_path": "autograd_grad",
        "activation.recompute": "checkpoint_non_reentrant_by_layer",
        **checkpoint_fields(),
    }
    candidate = vpx.Candidate(
        "family",
        "checkpoint-row",
        settings,
        admission_status="passed",
    )

    def adapter_operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        def function(value: torch.Tensor) -> torch.Tensor:
            return value * 3.0

        return vpx.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="activation.recompute",
        )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        output = adapter_operation_factory(candidate, batch, vector)()
        assert isinstance(vector, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        assert torch.equal(output, vector * 3.0)

        return reference_passed()

    runtime = runtime_config(
        candidates=(candidate,),
        operation_factory=adapter_operation_factory,
        reference_check=reference_check,
        materializer=materialize_candidate,
        axis_registry=None,
        signature={"runtime": "checkpoint-adapter"},
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=one_call_cpu_target(),
        runtime=runtime,
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "checkpoint-row"

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert not buffers
        assert batch
        assert context.family == "family"

        return params["weight"].sum()

    standard_factory = vpx.standard_operation_factory(
        ops.gradient("family", "objective", aggregation="sum"),
        params=dict(model.named_parameters()),
        buffers={},
        scalar_objectives={"objective": scalar_objective},
    )

    with pytest.raises(vp.MaterializationError):
        standard_factory(candidate, {"family": "family"}, torch.tensor([1.0]))()


def test_transformers_attention_admission_axis() -> None:
    policy = vpa.TransformersAttentionPolicy(
        model_config_hash="model",
        use_cache=False,
        softcap={"logit_softcap": 30.0},
        mask_semantics="boolean_keep_mask",
        causal_policy="causal",
        backend_numeric_policy={"backend": "flash_attention_2"},
        determinism={"deterministic": True},
        padding_limit=4096,
        forced_kernel_available=True,
    )
    axis = vpa.transformers_attention_axis(
        (
            "transformers_eager",
            "transformers_sdpa",
            "transformers_flash_attention_2",
            "transformers_flash_attention_3",
            "transformers_flash_attention_4",
            "paged|flash_attention_2",
            "paged|flash_attention_3",
            "paged|flash_attention_4",
        ),
        policy=policy,
    )
    flash = vpx.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_flash = vpx.Candidate(
        "family",
        "float-flash",
        {
            "attention.frontend": "transformers_flash_attention_2",
            "dtype.model_compute": "fp32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash3 = vpx.Candidate(
        "family",
        "flash3",
        {
            "attention.frontend": "transformers_flash_attention_3",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    flash4 = vpx.Candidate(
        "family",
        "flash4",
        {
            "attention.frontend": "transformers_flash_attention_4",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    paged_flash = vpx.Candidate(
        "family",
        "paged-flash",
        {
            "attention.frontend": "paged|flash_attention_4",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    attentions = vpx.Candidate(
        "family",
        "attentions",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": True,
        },
    )
    math_attention = vpx.Candidate(
        "family",
        "math",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    math_attentions = vpx.Candidate(
        "family",
        "math-attentions",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "math",
            "module_mode": "eval",
            "dropout_p": 0.0,
            "output_attentions": True,
        },
    )
    direct_flash_kernel = vpx.Candidate(
        "family",
        "sdpa-flash",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "flash_attention",
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    float_direct_flash_kernel = vpx.Candidate(
        "family",
        "float-sdpa-flash",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "flash_attention",
            "dtype.model_compute": "fp32",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    priority_sdpa = vpx.Candidate(
        "family",
        "priority-sdpa",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("flash_attention", "math"),
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    bad_priority_sdpa = vpx.Candidate(
        "family",
        "bad-priority-sdpa",
        {
            "attention.frontend": "transformers_sdpa",
            "attention.sdpa_kernel": "priority_list",
            "attention.sdpa_priority_list": ("priority_list",),
            "dtype.model_compute": "bf16",
            "module_mode": "eval",
            "dropout_p": 0.0,
        },
    )
    eval_dropout = vpx.Candidate(
        "family",
        "dropout",
        {
            "attention.frontend": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.1,
        },
    )
    assert axis.admit(flash) == (True, None)
    assert axis.admit(float_flash)[0] is False
    assert axis.admit(flash3) == (True, None)
    assert axis.admit(flash4) == (True, None)
    assert axis.admit(paged_flash) == (True, None)
    assert axis.admit(attentions)[0] is False
    assert axis.admit(math_attention) == (True, None)
    assert axis.admit(math_attentions)[0] is False
    assert axis.admit(direct_flash_kernel) == (True, None)
    assert axis.admit(float_direct_flash_kernel)[0] is False
    assert axis.admit(priority_sdpa) == (True, None)
    assert axis.admit(bad_priority_sdpa)[0] is False
    assert axis.admit(eval_dropout)[0] is False
    assert axis.signature()["identity"]["softcap"] == {"logit_softcap": 30.0}

    no_padding_policy = dataclasses.replace(policy, padding_limit=None)
    no_padding_axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=no_padding_policy,
    )
    unavailable_policy = dataclasses.replace(
        policy,
        forced_kernel_available=False,
        forced_kernel_failure_reason="kernel unavailable",
    )
    unavailable_axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=unavailable_policy,
    )

    assert no_padding_axis.admit(flash)[0] is False
    assert unavailable_axis.admit(flash) == (False, "kernel unavailable")
    assert (
        vpa.admit_transformers_attention(
            vpx.Candidate(
                "family",
                "unknown",
                {"attention.frontend": "unknown"},
            ),
            policy=policy,
        )[0]
        is False
    )

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attention_axis(("unknown",), policy=policy)


def test_axis_registry_admits_grid_and_records_failed_admission() -> None:
    model = torch.nn.Linear(1, 1)
    registry = vpx.AxisRegistry()

    def admit_dtype(candidate: vpx.Candidate) -> tuple[bool, str | None]:
        if candidate.settings["dtype"] == "fp16":
            return False, "float16 disabled"

        return True, None

    registry.register(
        vpx.AxisDescriptor(
            "dtype",
            ("dtype",),
            ("fp32", "fp16"),
            admission_rule=admit_dtype,
        )
    )

    with pytest.raises(vp.AdmissionError):
        registry.register(vpx.AxisDescriptor("other", ("dtype",), ("float64",)))

    candidates = vpx.settings_product(
        "family",
        {"dtype": ("fp32", "fp16")},
        generator_id="grid",
        generator_version="1",
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.settings["dtype"] == "fp32"
        assert batch["family"] == "family"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.settings["dtype"] == "fp32"
        assert batch["family"] == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=one_call_cpu_target(),
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            registry,
            {"generator": "grid"},
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert tuple(candidate.candidate_id for candidate in candidates) == (
        "family:0",
        "family:1",
    )
    assert plan.selected["family"].settings == {"dtype": "fp32"}
    assert plan.full_size_records[1].status == "failed"
    assert plan.full_size_records[1].error_type == "AdmissionError"

    invalid = vpx.Candidate(
        "family",
        "invalid",
        {"dtype": "float64"},
        changed_axes=("dtype",),
    )

    assert registry.admit(invalid).admission_status == "failed"


def test_tune_records_runtime_full_size_check_metadata() -> None:
    class PassingFullSizeCheck:
        def __init__(self) -> None:
            self.calls = []

        @staticmethod
        def identity() -> Mapping[str, object]:
            return {"full_size_check": "tests.passing"}

        def __call__(
            self,
            candidate: vpx.Candidate,
            inputs: tuple[tuple[vpx.Batch, vpx.TensorTree], ...],
            output: vpx.TensorTree,
            samples: tuple[vpx.Measurement, ...],
        ) -> Mapping[str, object]:
            assert candidate.candidate_id == "flash"
            assert len(inputs) == 1
            assert isinstance(output, tuple)
            assert len(samples) == 1
            self.calls.append(candidate.candidate_id)

            return {"full_size_agreement_passed": True}

    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
        admission_status="passed",
    )
    full_size_check = PassingFullSizeCheck()

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "flash"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "flash"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=dataclasses.replace(
            one_call_cpu_target(),
            allowed_attention_frontends=("pytorch_sdpa_direct",),
            allowed_sdpa_kernels=("flash_attention",),
        ),
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "full-size-check"},
            full_size_check=full_size_check,
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert full_size_check.calls == ["flash"]
    assert problem.runtime.identity()["full_size_check"] == {
        "full_size_check": "tests.passing"
    }
    assert plan.selected["family"].candidate_id == "flash"
    assert (
        plan.records["family"].selection_metadata["full_size_agreement_passed"] is True
    )


def test_settings_product_expands_registry_multi_key_axis() -> None:
    registry = vpx.AxisRegistry()
    registry.register(
        vpx.AxisDescriptor(
            "pair",
            ("left", "right"),
            ({"left": 1, "right": 2},),
        )
    )
    candidates = vpx.settings_product(
        "family",
        {"pair": ({"left": 1, "right": 2},)},
        axis_registry=registry,
    )

    assert candidates[0].settings == {"left": 1, "right": 2}
    assert candidates[0].changed_axes == ("pair",)
    assert registry.admit(candidates[0]).admission_status == "passed"

    with pytest.raises(vp.AdmissionError):
        vpx.settings_product(
            "family",
            {"pair": ({"left": 1},)},
            axis_registry=registry,
        )


def test_standard_axis_registry_validates_core_axes() -> None:
    registry = vpx.standard_axis_registry()
    torch_func_fields = {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": True,
        "forward_ad_supported": True,
    }
    compile_axis_settings = {
        "metric.multiply_path": "dense_matmul",
        "compile.enabled": "true",
        "compile.backend": "inductor",
        "compile.mode": "default",
        "compile.fullgraph": "false",
        "compile.dynamic": None,
        "compile.compiled_autograd": "false",
        "compile.options.epilogue_fusion": "false",
        "compile.options.shape_padding": "false",
        "compile.cuda_graphs": "false",
        "compile.cache_state": "cold_compile",
    }
    vector_vmap_fields = {
        "vectorization.mode": "vmap",
        "vectorization.vmap_chunk_size": 2,
    }

    def candidate(candidate_id: str, settings: Mapping[str, object]) -> vpx.Candidate:
        return vpx.Candidate("family", candidate_id, settings)

    def torch_func_settings(settings: Mapping[str, object]) -> dict[str, object]:
        return {**torch_func_fields, **settings}

    def vector_vmap_settings(
        settings: Mapping[str, object],
        in_dims: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            **torch_func_fields,
            **vector_vmap_fields,
            "vectorization.in_dims": dict(in_dims),
            **settings,
        }

    cases = (
        (
            "row",
            {
                "dtype.model_compute": "bf16",
                "hvp.path": "reverse_over_reverse",
                "numeric.float32_matmul_precision": "high",
            },
            "passed",
        ),
        ("bad-flag", {"numeric.bf16_reduced_precision_reduction": True}, "failed"),
        ("bad-dtype", {"dtype.model_compute": "float64"}, "failed"),
        ("fp8-storage", {"dtype.parameter_storage": "fp8_when_supported"}, "passed"),
        ("fp8-compute", {"dtype.model_compute": "fp8_when_supported"}, "passed"),
        ("bad-path", {"hvp.path": "reverse_over_forward"}, "failed"),
        ("missing-torch-func-fields", {"jvp.path": "torch_func_jvp"}, "failed"),
        (
            "valid-torch-func",
            torch_func_settings({"jvp.path": "torch_func_jvp"}),
            "passed",
        ),
        (
            "valid-ggn-linearize",
            torch_func_settings({
                "ggn.jvp_path": "torch_func_linearize",
                "ggn.vjp_path": "torch_func_vjp",
            }),
            "passed",
        ),
        (
            "missing-ggn-linearize-fields",
            {
                "ggn.jvp_path": "torch_func_linearize",
                "ggn.vjp_path": "torch_func_vjp",
            },
            "failed",
        ),
        (
            "valid-forward-ad",
            {
                "jvp.path": "forward_ad_dual",
                "requires_forward_ad": True,
                "forward_ad_supported": True,
            },
            "passed",
        ),
        (
            "unsupported-forward-ad",
            {
                "jvp.path": "forward_ad_dual",
                "requires_forward_ad": True,
                "forward_ad_supported": False,
            },
            "failed",
        ),
        (
            "valid-vmap",
            torch_func_settings({
                "empirical_fisher.grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "schedule.per_example": "vmap",
                "batch.empirical_example_batch_size": 2,
            }),
            "passed",
        ),
        (
            "valid-sampled-vmap",
            torch_func_settings({
                "sampled_fisher.score_grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "schedule.per_example": "vmap",
                "batch.fisher_sample_batch_size": 2,
            }),
            "passed",
        ),
        (
            "valid-hvp-vmap",
            vector_vmap_settings(
                {
                    "hvp.path": "linearize_grad",
                    "requires_forward_ad": True,
                    "forward_ad_supported": True,
                },
                {"w": 0},
            ),
            "passed",
        ),
        (
            "valid-jvp-vmap",
            vector_vmap_settings(
                {
                    "jvp.path": "torch_func_jvp",
                    "requires_forward_ad": True,
                    "forward_ad_supported": True,
                },
                {"w": 0},
            ),
            "passed",
        ),
        (
            "valid-vjp-vmap",
            vector_vmap_settings(
                {
                    "vjp.path": "torch_func_vjp",
                    "requires_forward_ad": False,
                },
                {"y": 0},
            ),
            "passed",
        ),
        (
            "valid-ggn-vmap",
            vector_vmap_settings(
                {
                    "ggn.jvp_path": "torch_func_jvp",
                    "ggn.vjp_path": "torch_func_vjp",
                    "requires_forward_ad": True,
                    "forward_ad_supported": True,
                },
                {"w": 0},
            ),
            "passed",
        ),
        (
            "valid-fisher-vector-vmap",
            vector_vmap_settings(
                {
                    "fisher.expectation_path": "explicit_full_expectation_score_rows",
                    "fisher.accumulation": "materialize_score_gradients",
                    "requires_forward_ad": False,
                    "forward_ad_supported": False,
                },
                {"w": 0},
            ),
            "passed",
        ),
        (
            "valid-composition-vmap",
            vector_vmap_settings(
                {
                    "composition.execution": "stream_child_outputs",
                    "requires_forward_ad": False,
                    "forward_ad_supported": False,
                },
                {"w": 0},
            ),
            "passed",
        ),
        (
            "missing-vmap-randomness",
            {
                "fisher.expectation_path": "explicit_full_expectation_score_rows",
                "fisher.accumulation": "materialize_score_gradients",
                "vectorization.mode": "vmap",
                "vectorization.vmap_chunk_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            "failed",
        ),
        ("stray-vmap-randomness", {"vectorization.randomness": "same"}, "failed"),
        (
            "valid-manual-batch",
            {
                "hvp.path": "reverse_over_reverse",
                "vectorization.mode": "manual_batch",
                "vectorization.batch_size": 2,
                "vectorization.in_dims": {"w": 0},
            },
            "passed",
        ),
        (
            "rejected-ggn-vmap-autograd-vjp",
            vector_vmap_settings(
                {
                    "ggn.jvp_path": "torch_func_jvp",
                    "ggn.vjp_path": "autograd_grad_outputs",
                    "requires_forward_ad": True,
                    "forward_ad_supported": True,
                },
                {"w": 0},
            ),
            "failed",
        ),
        (
            "rejected-forward-ad-jvp-vmap",
            {
                "jvp.path": "forward_ad_dual",
                "requires_forward_ad": True,
                "forward_ad_supported": True,
                **vector_vmap_fields,
                "vectorization.in_dims": {"w": 0},
            },
            "failed",
        ),
        (
            "valid-hvp-single-loop",
            {
                "hvp.path": "reverse_over_reverse",
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"w": 0},
            },
            "passed",
        ),
        (
            "missing-sampled-vmap-schedule",
            torch_func_settings({
                "sampled_fisher.score_grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "batch.fisher_sample_batch_size": 2,
            }),
            "failed",
        ),
        (
            "stray-vmap-in-dims",
            torch_func_settings({
                "empirical_fisher.grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "schedule.per_example": "vmap",
                "batch.empirical_example_batch_size": 2,
                "vectorization.in_dims": {"x": 0},
            }),
            "failed",
        ),
        (
            "invalid-vmap-in-dims",
            vector_vmap_settings(
                {
                    "hvp.path": "linearize_grad",
                    "requires_forward_ad": True,
                    "forward_ad_supported": True,
                },
                {"x": "0"},
            ),
            "failed",
        ),
        (
            "forward-ad-vmap",
            torch_func_settings({
                "empirical_fisher.grad_path": "vmap_grad",
                "schedule.per_example": "vmap",
                "batch.empirical_example_batch_size": 2,
            }),
            "failed",
        ),
        (
            "missing-vmap-schedule",
            torch_func_settings({
                "empirical_fisher.grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "batch.empirical_example_batch_size": 2,
            }),
            "failed",
        ),
        (
            "valid-manual-per-example",
            {
                "fisher.score_grad_path": "torch_autograd_grad_loop",
                "schedule.per_example": "manual_batch",
                "batch.fisher_sample_batch_size": 2,
            },
            "passed",
        ),
        (
            "missing-manual-per-example-size",
            {
                "sampled_fisher.score_grad_path": "torch_autograd_grad_loop",
                "schedule.per_example": "manual_batch",
            },
            "failed",
        ),
        (
            "single-loop-vmap",
            torch_func_settings({
                "empirical_fisher.grad_path": "vmap_grad",
                "requires_forward_ad": False,
                "schedule.per_example": "vmap",
                "vectorization.mode": "single_loop",
                "vectorization.in_dims": {"x": 0, "normalization": None},
            }),
            "passed",
        ),
        (
            "vmap-without-vmap-path",
            {
                "empirical_fisher.grad_path": "torch_autograd_grad_loop",
                "vectorization.mode": "vmap",
            },
            "failed",
        ),
        ("manual-batch", {"vectorization.mode": "manual_batch"}, "failed"),
        (
            "valid-microbatch-accumulation",
            {
                "gradient.path": "torch_autograd_grad",
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            "passed",
        ),
        (
            "missing-microbatch-size",
            {
                "gradient.path": "torch_autograd_grad",
                "schedule.gradient_accumulation": "microbatch_accumulate",
            },
            "failed",
        ),
        ("stray-microbatch-size", {"batch.data_microbatch_size": 2}, "failed"),
        (
            "jvp-microbatch",
            torch_func_settings({
                "jvp.path": "torch_func_jvp",
                "requires_forward_ad": True,
                "forward_ad_supported": True,
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            }),
            "passed",
        ),
        (
            "hvp-microbatch",
            {
                "hvp.path": "reverse_over_reverse",
                "schedule.gradient_accumulation": "microbatch_accumulate",
                "batch.data_microbatch_size": 2,
            },
            "passed",
        ),
        (
            "valid-input-memory-axes",
            {
                "schedule.per_token": "loop",
                "input.batch_layout": "dense_padded",
                "input.length_grouping": "none",
                "input.host_to_device": "outside_measured_call",
                "input.residency": "cpu_staged",
                "teacher_outputs": "precomputed_cpu",
                "memory.vector_residency": "cpu_staged",
                "memory.factor_residency": "cpu_staged",
                "memory.output_buffers": "fresh_allocation",
            },
            "passed",
        ),
        (
            "valid-package-runtime-axes",
            {
                "dtype.autodiff_compute": "bf16",
                "dtype.accumulation": "fp32",
                "gradient.graph_schedule": "build_once",
                "memory.primal_outputs": "retain",
                "memory.jvp_outputs": "retain",
                "memory.output_cotangents": "retain",
                "activation.recompute": "checkpoint_selective",
                "activation.offload": "custom_saved_tensor_hooks",
                "activation.pack_hook": "pack",
                "activation.unpack_hook": "unpack",
                "checkpoint.use_reentrant": "false",
                "checkpoint.early_stop": "true",
                "checkpoint.preserve_rng_state": "false",
                "checkpoint.determinism_check": "default",
                "checkpoint.context_fn": "declared_context_pair",
                "checkpoint.context_fn_callable": "default",
                "checkpoint.moves_to_new_device": "false",
                "checkpoint.uses_global_state": "false",
                "metric.block_schedule": "layer_blocks",
                "inverse_metric.block_schedule": "module_blocks",
                "fusion.norm": "model_default",
                "fusion.mlp": "model_default",
                "fusion.rope": "model_default",
                "fusion.logits": "model_default",
                "fusion.loss": "model_default",
            },
            "passed",
        ),
        (
            "invalid-package-runtime-axis",
            {"checkpoint.use_reentrant": "true"},
            "failed",
        ),
        (
            "callable-activation-hook-axis",
            {
                "activation.offload": "custom_saved_tensor_hooks",
                "activation.pack_hook": len,
                "activation.unpack_hook": len,
            },
            "failed",
        ),
        (
            "callable-checkpoint-context-axis",
            {
                "checkpoint.context_fn": "declared_context_pair",
                "checkpoint.context_fn_callable": contextlib.nullcontext,
            },
            "failed",
        ),
        (
            "valid-compile-boundary",
            {**compile_axis_settings, "compile.boundary": "metric_multiply"},
            "passed",
        ),
        (
            "valid-bound-operator-boundary",
            {
                **compile_axis_settings,
                "compile.boundary": "bound_operator_vector_step",
            },
            "passed",
        ),
        (
            "invalid-compile-boundary",
            {**compile_axis_settings, "compile.boundary": "unknown_boundary"},
            "failed",
        ),
        (
            "invalid-bound-operator-kind-boundary",
            {
                "gradient.path": "autograd_grad",
                "compile.enabled": "true",
                "compile.boundary": "bound_operator_vector_step",
                "compile.backend": "inductor",
                "compile.mode": "default",
                "compile.fullgraph": "false",
                "compile.dynamic": None,
                "compile.compiled_autograd": "false",
                "compile.options.epilogue_fusion": "false",
                "compile.options.shape_padding": "false",
                "compile.cuda_graphs": "false",
                "compile.cache_state": "warm_cache",
            },
            "failed",
        ),
        (
            "valid-chunk-axes",
            {
                "batch.hvp_row_batch_size": 2,
                "batch.ggn_batch_size": 3,
                "chunk.token_block_size": 4,
                "chunk.sequence_position_block_size": 5,
                "chunk.output_cotangent_block_size": 6,
                "chunk.parameter_block_size": 7,
                "chunk.layer_block_size": 8,
                "chunk.lm_head_weight_chunk_bytes": 9,
            },
            "passed",
        ),
        ("invalid-chunk-axis", {"chunk.token_block_size": 0}, "failed"),
    )

    for candidate_id, settings, expected_status in cases:
        admitted = registry.admit(candidate(candidate_id, settings))
        assert admitted.admission_status == expected_status

    assert (
        registry.axes["dtype.model_compute"].admit(
            candidate("bad-dtype", {"dtype.model_compute": "float64"})
        )[0]
        is False
    )

    with pytest.raises(vp.AdmissionError):
        vpx.AxisRegistry().register(vpx.AxisDescriptor("bad", ("x",), ()))


def test_standard_axis_registry_accepts_concrete_registered_compile_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.compiler, "list_backends", lambda: ["custom_backend"])
    registry = vpx.standard_axis_registry()
    concrete = vpx.Candidate(
        "family",
        "concrete-backend",
        {"compile.backend": "custom_backend"},
    )
    placeholder = vpx.Candidate(
        "family",
        "placeholder-backend",
        {"compile.backend": "registered_backend"},
    )
    missing = vpx.Candidate(
        "family",
        "missing-backend",
        {"compile.backend": "missing_backend"},
    )

    assert registry.admit(concrete).admission_status == "passed"
    assert registry.admit(placeholder).admission_status == "failed"
    assert registry.admit(missing).admission_status == "failed"


def test_problem_signature_includes_axis_registry_identity() -> None:
    model = torch.nn.Linear(1, 1)
    first_registry = vpx.AxisRegistry()
    second_registry = vpx.AxisRegistry()
    first_registry.register(
        vpx.AxisDescriptor(
            "axis",
            ("axis",),
            ("value",),
            adapter_id="adapter",
            adapter_version="1",
        )
    )
    second_registry.register(
        vpx.AxisDescriptor(
            "axis",
            ("axis",),
            ("value",),
            adapter_id="adapter",
            adapter_version="2",
        )
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate
        assert batch
        assert vector

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate
        assert batch

        return vpx.constant_operation(vector)

    first = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=runtime_config(
            (),
            operation_factory,
            reference_check,
            materialize_candidate,
            first_registry,
            {"generator": "registry"},
        ),
    )
    second = dataclasses.replace(
        first,
        runtime=dataclasses.replace(first.runtime, axis_registry=second_registry),
    )

    assert first_registry.signature() != second_registry.signature()
    assert first.input_signature() != second.input_signature()


def test_package_metadata_matches_current_pypa_fields() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    sdist_include = set(
        pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )

    assert project["version"] == PACKAGE_VERSION
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["import-names"] == ["vptune"]
    assert "numpy" in project["dependencies"]
    assert "torch>=2.12,<2.13" in project["dependencies"]
    assert not any(
        classifier.startswith("License ::") for classifier in project["classifiers"]
    )
    assert pyproject["build-system"]["requires"] == ["hatchling==1.30.1"]
    assert (
        pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["core-metadata-version"]
        == "2.4"
    )
    assert (
        pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["core-metadata-version"]
        == "2.4"
    )
    assert "/.github" in sdist_include
    assert "/SECURITY.md" in sdist_include


def test_repository_security_files_are_declared() -> None:
    root = Path(__file__).parents[1]
    security = (root / "SECURITY.md").read_text(encoding="utf-8")
    owners = (root / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    dependabot = (root / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert "https://github.com/bmucsanyi/vptune/security/advisories/new" in security
    assert "Do not report suspected vulnerabilities in public issues" in security
    assert owners == "* @bmucsanyi\n"
    assert 'package-ecosystem: "uv"' in dependabot
    assert 'package-ecosystem: "pre-commit"' in dependabot


def test_package_owned_identities_use_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidates_module, "PACKAGE_VERSION", "9.9.9")
    monkeypatch.setattr(attention_module, "PACKAGE_VERSION", "9.9.9")
    monkeypatch.setattr(measure_module, "PACKAGE_VERSION", "9.9.9")
    monkeypatch.setattr(runtime_module, "PACKAGE_VERSION", "9.9.9")

    def scalar_objective(
        params: vpx.ParameterTree,
        buffers: vpx.BufferTree,
        batch: vpx.Batch,
        context: vpx.ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert batch == {}
        assert context.family == "gradient"

        return params["w"].pow(2).sum()

    runtime = vpx.standard_runtime_config(
        ops.gradient("gradient", "loss", aggregation="sum"),
        params={"w": torch.tensor([1.0], dtype=torch.float64)},
        buffers={},
        candidates=(
            vpx.Candidate(
                "gradient",
                "row",
                {"operator_path": "autograd_grad"},
                admission_status="passed",
            ),
        ),
        thresholds={
            "max_abs_diff": 1e-12,
            "max_rel_diff": 1e-12,
            "directional_abs_diff": 1e-12,
            "directional_rel_diff": 1e-12,
        },
        objective_signature={"loss": "versioned"},
        axis_registry=None,
        scalar_objectives={"loss": scalar_objective},
    )
    grid = candidates_module.settings_product("family", {"axis": ("value",)})

    assert candidates_module.axis_table().package_version == "9.9.9"
    assert vpx.AxisDescriptor("axis", ("axis",), ("value",)).adapter_version == "9.9.9"
    assert grid[0].generator_version == "9.9.9"
    assert attention_module.core_attention_axis().adapter_version == "9.9.9"
    assert CPUMemoryBackend.identity()["backend_version"] == "9.9.9"
    assert (
        measure_module.CUDAMemoryBackend(("cuda:0",)).identity()["backend_version"]
        == "9.9.9"
    )
    assert runtime.materializer.identity()["materializer_version"] == "9.9.9"


def test_settings_product_rejects_empty_axis_values() -> None:
    with pytest.raises(vp.AdmissionError, match="candidate axis has no values: axis"):
        candidates_module.settings_product("family", {"axis": ()})


def test_target_admission_rejects_disallowed_settings() -> None:
    model = torch.nn.Linear(1, 1)
    candidates = (
        vpx.Candidate(
            "family",
            "allowed",
            {
                "dtype.model_compute": "fp32",
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "math",
            },
            admission_status="passed",
        ),
        vpx.Candidate(
            "family",
            "blocked",
            {"dtype.model_compute": "bf16"},
            admission_status="passed",
        ),
        vpx.Candidate(
            "family",
            "blocked-frontend",
            {
                "dtype.model_compute": "fp32",
                "attention.frontend": "transformers_flash_attention_2",
            },
            admission_status="passed",
        ),
        vpx.Candidate(
            "family",
            "blocked-kernel",
            {
                "dtype.model_compute": "fp32",
                "attention.frontend": "pytorch_sdpa_direct",
                "attention.sdpa_kernel": "flash_attention",
            },
            admission_status="passed",
        ),
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "allowed"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "allowed"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    target = vpx.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("fp32",),
        allowed_attention_frontends=("pytorch_sdpa_direct",),
        allowed_sdpa_kernels=("math",),
        allowed_sharding_modes=("single_device",),
        timing_policy=vpx.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        ),
        selection_policy=vpx.SelectionPolicy(),
        search_policy=vpx.SearchPolicy(strategy="exhaustive"),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "target-admission"},
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "allowed"
    failed_rows = tuple(row for row in plan.full_size_records if row.status == "failed")
    assert len(failed_rows) == 3
    assert all(row.error_type == "AdmissionError" for row in failed_rows)


def test_autobatch_bridge_selects_candidate_by_positive_index_domain() -> None:
    calls = []
    candidates = (
        vpx.Candidate("family", "first", {}, admission_status="passed"),
        vpx.Candidate("family", "second", {}, admission_status="passed"),
    )

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert values == (1, 2)
        assert cache_key == ("case", "bridge")
        assert warmup_steps == 1
        assert measure_steps == 2
        assert devices == [0]
        assert goal == autobatch.Goal.fastest_step()
        probe(2)

        return 2

    selected = vpx.select_fastest_candidate_with_autobatch(
        candidates,
        lambda candidate: calls.append(candidate.candidate_id),
        cache_key=("case", "bridge"),
        warmup_steps=1,
        measure_steps=2,
        devices=[0],
        find=fake_find,
    )

    assert selected.candidate_id == "second"
    assert calls == ["second"]


AUTOBATCH_FAILURE_SIGNALS = (
    "backend_rejection",
    "oom",
    "reference_failure",
    "runtime_failure",
)


def make_autobatch_domain(
    *,
    min_value: int = 1,
    max_value: int = 4,
    initial_value: int = 1,
    growth: str = "doubling",
    values: tuple[int, ...] = (1, 2, 4),
    settings_by_value: Mapping[int, Mapping[str, object]] | None = None,
    objective: str = "fastest_passing",
    failure_signals: tuple[str, ...] = AUTOBATCH_FAILURE_SIGNALS,
    termination: str = "exhausted_declared_values",
    cache_key_case: str = "autobatch-domain",
) -> vpx.AutobatchDomain:
    selected_settings = (
        {value: {"batch_size": value} for value in values}
        if settings_by_value is None
        else settings_by_value
    )

    return vpx.AutobatchDomain(
        axis_name="batch_size",
        min_value=min_value,
        max_value=max_value,
        initial_value=initial_value,
        growth=growth,
        values=values,
        settings_by_value=selected_settings,
        value_to_settings_id="tests.batch_size_settings",
        admission_identity={"case": "test"},
        objective=objective,
        failure_signals=failure_signals,
        termination=termination,
        warmup_steps=0,
        measure_steps=1,
        devices=(0,),
        cache_key_payload={"case": cache_key_case},
    )


def autobatch_problem(
    *,
    model: torch.nn.Module,
    target: vpx.Target,
    operation_factory: vpx.OperationFactoryCallback,
    reference_check: vpx.ReferenceCheckCallback,
    generator: str,
    domain: vpx.AutobatchDomain,
) -> vpx.Problem:
    return vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            (vpx.Candidate("family", "base", {}, admission_status="passed"),),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": generator},
            (domain,),
        ),
    )


def test_autobatch_domain_rejects_invalid_finite_search_shape() -> None:
    with pytest.raises(RuntimeError, match="strictly increasing"):
        make_autobatch_domain(values=(1, 1, 2))

    with pytest.raises(RuntimeError, match="doubling"):
        make_autobatch_domain(values=(1, 3, 4))

    with pytest.raises(RuntimeError, match="initial_value"):
        make_autobatch_domain(initial_value=2)

    with pytest.raises(RuntimeError, match="settings_by_value"):
        make_autobatch_domain(settings_by_value={1: {"batch_size": 1}})

    with pytest.raises(RuntimeError, match="objective"):
        make_autobatch_domain(objective="fastest_step")

    with pytest.raises(RuntimeError, match="bracketed_failure_frontier"):
        make_autobatch_domain(
            objective="largest_passing",
            termination="exhausted_declared_values",
        )


def test_tune_fast_strategy_delegates_autobatch_domain_to_autobatch_find(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    find_calls = []
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.settings["batch_size"] in {1, 2}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(candidate.candidate_id)

            return vector * float(candidate.settings["batch_size"])

        return operation

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert values == (1, 2)
        assert goal == autobatch.Goal.fastest_step()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        find_calls.append(tuple(values))
        probe(1)
        probe(2)

        return 2

    monkeypatch.setattr(autobatch_bridge.autobatch, "find", fake_find)

    target = dataclasses.replace(
        one_call_cpu_target(),
        search_policy=vpx.SearchPolicy(strategy="fast"),
    )
    problem = autobatch_problem(
        model=model,
        target=target,
        operation_factory=operation_factory,
        reference_check=reference_check,
        generator="autobatch-domain",
        domain=make_autobatch_domain(
            max_value=2,
            growth="linear_step",
            values=(1, 2),
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0, 2.0, 3.0)),
    )

    assert find_calls == [(1, 2)]
    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base|batch_size=1",
        "base|batch_size=2",
    )
    assert calls == ["base|batch_size=1", "base|batch_size=2"]
    assert plan.selected["family"].candidate_id == "base|batch_size=2"
    assert plan.selected["family"].settings["batch_size"] == 2


def test_tune_reuses_current_autobatch_rows_on_next_tune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    find_calls = []
    reference_calls = []
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.settings["batch_size"] in {1, 2}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        reference_calls.append(candidate.candidate_id)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(candidate.candidate_id)

            return vector * float(candidate.settings["batch_size"])

        return operation

    def first_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert goal == autobatch.Goal.fastest_step()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        find_calls.append(tuple(values))
        probe(1)
        probe(2)

        return 2

    def second_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert probe
        assert values
        assert goal
        assert cache_key
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        message = "saved autobatch rows should bypass find"
        raise AssertionError(message)

    target = dataclasses.replace(
        one_call_cpu_target(),
        search_policy=vpx.SearchPolicy(strategy="fast"),
    )
    problem = autobatch_problem(
        model=model,
        target=target,
        operation_factory=operation_factory,
        reference_check=reference_check,
        generator="autobatch-resume",
        domain=make_autobatch_domain(
            max_value=2,
            growth="linear_step",
            values=(1, 2),
            cache_key_case="autobatch-resume",
        ),
    )

    monkeypatch.setattr(autobatch_bridge.autobatch, "find", first_find)
    first_plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0, 2.0, 3.0)),
    )
    monkeypatch.setattr(autobatch_bridge.autobatch, "find", second_find)
    second_plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock(()),
    )

    assert find_calls == [(1, 2)]
    assert calls == ["base|batch_size=1", "base|batch_size=2"]
    assert reference_calls == ["base|batch_size=1", "base|batch_size=2"]
    assert first_plan.selected["family"].candidate_id == "base|batch_size=2"
    assert second_plan.selected["family"].candidate_id == "base|batch_size=2"
    assert not (
        tmp_path / "full_size" / "family" / "base|batch_size=2" / "result-000001.json"
    ).exists()


def test_autobatch_domain_filters_reference_failures_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = []
    find_values = []
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        if candidate.settings["batch_size"] == 1:
            message = "reference rejected value"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            probed.append(candidate.settings["batch_size"])

            return vector * float(candidate.settings["batch_size"])

        return operation

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert goal == autobatch.Goal.largest_safe()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        find_values.append(tuple(values))
        probe(2)

        return 2

    monkeypatch.setattr(autobatch_bridge.autobatch, "find", fake_find)
    target = one_call_cpu_target()
    problem = autobatch_problem(
        model=model,
        target=target,
        operation_factory=operation_factory,
        reference_check=reference_check,
        generator="autobatch-domain",
        domain=make_autobatch_domain(
            max_value=2,
            growth="linear_step",
            values=(1, 2),
            objective="largest_passing",
            termination="bracketed_failure_frontier",
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0)),
    )

    failed = {
        record.candidate_id: record
        for record in plan.full_size_records
        if record.status == "failed"
    }

    assert find_values == [(2,)]
    assert probed == [2]
    assert plan.selected["family"].candidate_id == "base|batch_size=2"
    assert failed["base|batch_size=1"].error_type == "ReferenceFailed"


def test_plan_replay_preserves_autobatch_selected_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.settings["batch_size"] in {1, 2}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return vector * float(candidate.settings["batch_size"])

        return operation

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert values == (1, 2)
        assert goal == autobatch.Goal.largest_safe()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        probe(1)
        probe(2)

        return 2

    monkeypatch.setattr(autobatch_bridge.autobatch, "find", fake_find)
    target = one_call_cpu_target()
    problem = autobatch_problem(
        model=model,
        target=target,
        operation_factory=operation_factory,
        reference_check=reference_check,
        generator="autobatch-domain",
        domain=make_autobatch_domain(
            max_value=2,
            growth="linear_step",
            values=(1, 2),
            objective="largest_passing",
            termination="bracketed_failure_frontier",
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 11.0)),
    )

    assert plan.selected["family"].candidate_id == "base|batch_size=2"

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={"family": materialize_candidate},
    )

    assert replayed.selected["family"].candidate_id == "base|batch_size=2"


def test_plan_replay_rejects_unsupported_selection_policy(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.replay-policy"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    context = dataclasses.replace(
        replay_context_for_plan(plan),
        selection_policy=vpx.SelectionPolicy(speed_statistic="mean_elapsed_seconds"),
    )

    with pytest.raises(vp.VPTuneError, match="unsupported speed statistic"):
        vpx.plan_from_json(
            read_record(tmp_path / "summaries" / "tuning.json"),
            replay_context=context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=saved_candidate_rows(tmp_path, plan),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_rejects_changed_memory_backend_identity(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.memory-backend"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    changed_input = dict(plan.input_signature)
    changed_input["measurement"] = {
        "memory_backend": {"backend_id": "tests.changed_memory_backend"}
    }
    family_signatures = dict(replay_context_for_plan(plan).family_input_signatures)
    family_signatures["family"] = changed_input
    context = dataclasses.replace(
        replay_context_for_plan(plan),
        input_signature=changed_input,
        family_input_signatures=family_signatures,
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)

    assert plan.input_signature["measurement"]["memory_backend"] == dict(
        CPUMemoryBackend().identity()
    )

    with pytest.raises(vp.StaleRecordError, match="input signature"):
        vpx.plan_from_json(
            read_record(tmp_path / "summaries" / "tuning.json"),
            replay_context=context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=saved_candidate_rows(tmp_path, plan),
            materializers={"family": materialize_candidate},
        )


def test_tune_rejects_adapter_identity_that_contradicts_runtime() -> None:
    model = torch.nn.Linear(1, 1)

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            return torch.tensor([1.0])

        return operation

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=runtime_config(
            (vpx.Candidate("family", "row", {}, admission_status="passed"),),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {
                "runtime": "test.adapter-runtime",
                "adapter_id": "adapter.test",
                "adapter_version": "1",
            },
        ),
    )

    with pytest.raises(vp.MaterializationError, match="adapter identity"):
        tune_problem(problem, memory_backend=CPUMemoryBackend())


def test_tune_uses_explicit_candidates_and_reference_checks(tmp_path: Path) -> None:
    calls = {"slow": 0, "bad": 0, "fast": 0}
    model = torch.nn.Linear(1, 1)
    candidates = (
        vpx.Candidate(
            "family",
            "slow",
            {"scale": 1.0},
            admission_status="passed",
        ),
        vpx.Candidate(
            "family",
            "bad",
            {"scale": 0.0},
            admission_status="passed",
        ),
        vpx.Candidate(
            "family",
            "fast",
            {"scale": 2.0},
            admission_status="passed",
        ),
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert batch["family"] == "family"
        assert batch["source"] == "reference"
        assert batch["check"] == "tree_close"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        if candidate.candidate_id == "bad":
            message = "bad row"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["family"] == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls[candidate.candidate_id] += 1

            if candidate.candidate_id == "fast":
                return vector * 2.0

            return vector

        return operation

    target = one_call_cpu_target()
    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "unit_test"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 10.0, 10.0, 15.0)),
    )

    assert plan.selected["family"].candidate_id == "fast"
    assert len(plan.full_size_records) == 3
    assert calls == {"slow": 1, "bad": 0, "fast": 1}
    assert plan.full_size_records[1].status == "failed"
    assert not plan.full_size_records[1].reference_passed
    assert len(plan.check_records) == 3
    assert plan.check_records[0].status == "passed"
    assert plan.check_records[1].status == "failed"
    selected_operator = vp.materialize(plan, name="family")

    assert torch.equal(selected_operator(), torch.tensor([2.0]))

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vpx.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={"family": materialize_candidate},
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "fast"
        assert record.candidate_id == "fast"
        assert context.family == "family"
        assert torch.equal(context.selected(), torch.tensor([2.0]))

        return vpx.ReferenceResult(
            "selected_plan_validation",
            {"max_abs_diff": 1e-6},
            {"max_abs_diff": 0.0},
        )

    validation_records = vp.validate_plan(plan, {"family": validator})

    assert validation_records[0].status == "passed"
    assert validation_records[0].name == "selected_plan_validation"


def test_tune_builds_measured_operation_before_clock(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    events = []
    clock_values = iter((0.0, 1.0))

    def clock() -> float:
        events.append("clock")

        return next(clock_values)

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)
        events.append("build")

        def operation() -> torch.Tensor:
            events.append("run")

            return vector

        return operation

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=one_call_cpu_target(),
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "unit_test.build-before-clock"},
        ),
    )

    tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=clock,
    )

    assert events == ["build", "clock", "run", "clock"]


def test_tune_writes_admission_failure_rows_without_measurement(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    failed_candidate = vpx.Candidate(
        "family",
        "blocked",
        {},
        admission_status="failed",
        admission_error="blocked by admission",
    )

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=runtime_config(
            (failed_candidate,),
            operation_factory_never_runs,
            reference_check_never_runs,
            materialize_candidate,
            None,
            {"generator": "admission-failure"},
        ),
    )

    with pytest.raises(vp.NoPassedCandidateError):
        tune_problem(
            problem,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    candidate_row = read_record(
        tmp_path / "candidates" / "family" / "blocked" / "candidate.json"
    )
    check_row = read_record(
        tmp_path / "references" / "family" / "blocked" / "tree_close.json"
    )
    full_size_row = read_record(
        tmp_path / "full_size" / "family" / "blocked" / "result.json"
    )

    assert candidate_row["status"] == "failed"
    assert check_row["status"] == "failed"
    assert check_row["error_type"] == "AdmissionError"
    assert full_size_row["status"] == "failed"
    assert full_size_row["reference_passed"] is False


def test_reference_failed_rows_are_rechecked_on_next_tune(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")
    calls = {"reference": 0, "operation": 0}
    problem = single_row_tuning_problem(
        model=model,
        operator=ops.gradient("family", "loss", aggregation="sum"),
        target=one_call_cpu_target(),
        row=candidate,
        calls=calls,
        generator="reference-rerun",
        reference_failure_call=1,
    )

    with pytest.raises(vp.NoPassedCandidateError):
        tune_problem(
            problem,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    failed_reference = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close.json"
    )
    passed_reference = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close-000001.json"
    )
    failed_full_size = read_record(
        tmp_path / "full_size" / "family" / "row" / "result.json"
    )
    passed_full_size = read_record(
        tmp_path / "full_size" / "family" / "row" / "result-000001.json"
    )

    assert calls == {"reference": 2, "operation": 1}
    assert plan.selected_candidate().candidate_id == "row"
    assert failed_reference["status"] == "failed"
    assert failed_reference["error_type"] == "ReferenceFailed"
    assert passed_reference["status"] == "passed"
    assert failed_full_size["status"] == "failed"
    assert failed_full_size["error_type"] == "ReferenceFailed"
    assert passed_full_size["status"] == "passed"


def test_tune_reuses_current_run_dir_rows_on_next_tune(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")
    calls = {"reference": 0, "operation": 0}
    problem = single_row_tuning_problem(
        model=model,
        operator=ops.gradient("family", "loss", aggregation="sum"),
        target=one_call_cpu_target(),
        row=candidate,
        calls=calls,
        generator="resume",
    )

    first_plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    second_plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock(()),
    )

    assert calls == {"reference": 1, "operation": 1}
    assert first_plan.selected_candidate().candidate_id == "row"
    assert second_plan.selected_candidate().candidate_id == "row"
    assert not (
        tmp_path / "references" / "family" / "row" / "tree_close-000001.json"
    ).exists()
    assert not (
        tmp_path / "full_size" / "family" / "row" / "result-000001.json"
    ).exists()


def test_tune_reruns_when_saved_candidate_settings_differ(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    calls = {"reference": 0, "operation": 0}

    def problem_for(scale: float) -> vpx.Problem:
        candidate = vpx.Candidate(
            "family",
            "row",
            {"scale": scale},
            admission_status="passed",
        )

        return single_row_tuning_problem(
            model=model,
            operator=ops.gradient("family", "loss", aggregation="sum"),
            target=one_call_cpu_target(),
            row=candidate,
            calls=calls,
            generator="resume-stale",
        )

    first_plan = tune_problem(
        problem_for(1.0),
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    second_plan = tune_problem(
        problem_for(2.0),
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((1.0, 3.0)),
    )

    assert calls == {"reference": 2, "operation": 2}
    assert first_plan.selected_candidate().settings == {"scale": 1.0}
    assert second_plan.selected_candidate().settings == {"scale": 2.0}
    assert (
        tmp_path / "references" / "family" / "row" / "tree_close-000001.json"
    ).exists()
    assert (tmp_path / "full_size" / "family" / "row" / "result-000001.json").exists()


def test_tune_records_reference_runtime_failures() -> None:
    model = torch.nn.Linear(1, 1)
    candidates = (
        vpx.Candidate("family", "bad", {}, admission_status="passed"),
        vpx.Candidate("family", "good", {}, admission_status="passed"),
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        if candidate.candidate_id == "bad":
            message = "reference runtime failed"
            raise RuntimeError(message)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "good"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=one_call_cpu_target(),
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "reference-runtime"},
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "good"
    assert plan.check_records[0].status == "failed"
    assert plan.check_records[0].error_type == "RuntimeError"
    assert plan.full_size_records[0].status == "failed"
    assert not plan.full_size_records[0].reference_passed


def test_tune_writes_records_and_produced_rows_are_current(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    problem = single_row_tuning_problem(
        model=model,
        operator=ops.hvp("family", "objective", aggregation="sum"),
        target=one_call_cpu_target(),
        row=candidate,
        calls={"reference": 0, "operation": 0},
        generator="write-test",
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    full_size_row = read_record(
        tmp_path / "full_size" / "family" / "row" / "result.json"
    )
    reference_row = read_record(
        tmp_path / "references" / "family" / "row" / "tree_close.json"
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")

    assert read_record(tmp_path / "candidates" / "family" / "row" / "candidate.json")
    candidate_row = read_record(
        tmp_path / "candidates" / "family" / "row" / "candidate.json"
    )
    replayed_candidate = vpx.candidate_record_from_json(candidate_row)

    assert replayed_candidate == candidate

    changed_candidate_row = dict(candidate_row)
    changed_candidate_row["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(full_size_row),),
            check_records=(vpx.check_record_from_json(reference_row),),
            candidate_records=(changed_candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    assert record_current(
        full_size_row,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert record_current(
        reference_row,
        record_type="reference",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        check_name="tree_close",
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert vpx.plan_record_current(summary, plan)

    stale_summary = dict(summary)
    stale_summary["generator_version"] = "stale"

    assert not vpx.plan_record_current(stale_summary, plan)

    changed_full_size = dict(full_size_row)
    changed_full_size["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(changed_full_size),),
            check_records=(vpx.check_record_from_json(reference_row),),
            candidate_records=(candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    changed_reference = dict(reference_row)
    changed_reference["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=(vpx.full_size_record_from_json(full_size_row),),
            check_records=(vpx.check_record_from_json(changed_reference),),
            candidate_records=(candidate_row,),
            materializers={
                "family": materialize_candidate,
            },
            run_dir=tmp_path,
        )

    replayed = vpx.plan_from_json(
        summary,
        replay_context=replay_context_for_plan(plan),
        full_size_records=(vpx.full_size_record_from_json(full_size_row),),
        check_records=(vpx.check_record_from_json(reference_row),),
        candidate_records=(candidate_row,),
        materializers={
            "family": materialize_candidate,
        },
        run_dir=tmp_path,
    )
    replayed_operator = vp.materialize(replayed, name="family")

    assert torch.equal(replayed_operator(), torch.tensor([1.0]))


def test_tune_measures_every_probe_input() -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate("family", "row", {}, admission_status="passed")

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(batch["index"])

            return vector

        return operation

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=TwoProbeData(),
        operator=ops.hvp("family", "objective", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=one_call_cpu_target(),
        runtime=runtime_config(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "two-probe"},
        ),
    )
    plan = tune_problem(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert calls == [0, 1]
    assert plan.full_size_records[0].output_signature["type"] == "sequence"


def test_tree_reference_check_compares_candidate_to_anchor() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"mode": "same"},
        admission_status="passed",
    )
    bad_candidate = vpx.Candidate(
        "family",
        "bad",
        {"mode": "bad"},
        admission_status="passed",
    )

    def anchor_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.candidate_id == "anchor"
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        return vpx.constant_operation(vector)

    def candidate_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        if candidate.candidate_id == "bad":
            return vpx.constant_operation(vector * 2.0)

        return vpx.constant_operation(vector)

    check = vpx.tree_reference_check(
        anchor_factory=anchor_factory,
        candidate_factory=candidate_factory,
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        anchor_candidate_id="anchor",
    )
    result = check(candidate, {"family": "family"}, torch.tensor([1.0]))

    assert result.measurements == {"max_abs_diff": 0.0, "max_rel_diff": 0.0}

    with pytest.raises(vp.ReferenceFailedError):
        check(bad_candidate, {"family": "family"}, torch.tensor([1.0]))


def test_tree_reference_check_rejects_self_anchor_candidate_id() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"mode": "same"},
        admission_status="passed",
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.family == "family"
        assert batch["family"] == "family"

        return vpx.constant_operation(vector)

    check = vpx.tree_reference_check(
        anchor_factory=operation_factory,
        candidate_factory=operation_factory,
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        anchor_candidate_id="row",
    )

    with pytest.raises(vp.MaterializationError, match="anchor candidate_id"):
        check(candidate, {"family": "family"}, torch.tensor([1.0]))


def test_records_round_trip_through_json(tmp_path: Path) -> None:
    input_signature = _input_signature("json")
    candidate = vpx.Candidate(
        "family",
        "row",
        {"dtype": "fp32"},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(2.0,),
        input_signature=input_signature,
    )
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )
    row_path = tmp_path / "full_size.json"
    plan_path = tmp_path / "plan.json"

    write_record(row_path, vpx.full_size_record_to_json(record))
    write_record(plan_path, vpx.plan_to_json(plan))

    loaded = read_record(row_path)
    round_tripped = vpx.full_size_record_from_json(loaded)
    loaded_plan = read_record(plan_path)

    assert round_tripped == record
    assert vpx.plan_record_current(loaded_plan, plan)

    stale_row = dict(loaded)
    stale_row["input_signature"] = {"case": "changed"}

    assert not record_current(
        stale_row,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=plan.input_signature,
        candidate_settings=candidate.settings,
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )

    stale_plan = dict(loaded_plan)
    stale_plan["input_signature"] = {"case": "changed"}

    assert not vpx.plan_record_current(stale_plan, plan)


def test_plan_replay_requires_all_saved_rows(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    problem = single_row_tuning_problem(
        model=model,
        operator=ops.hvp("family", "objective", aggregation="sum"),
        target=one_call_cpu_target(),
        row=candidate,
        calls={"reference": 0, "operation": 0},
        generator="replay-test",
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=(),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            run_dir=tmp_path,
        )

    mismatched_summary = dict(summary)
    mismatched_summary["records"] = {}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            mismatched_summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            run_dir=tmp_path,
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={},
            run_dir=tmp_path,
        )


def test_plan_replay_rejects_stale_context_and_materializer() -> None:
    input_signature = _input_signature("replay-context")
    other_signature = _input_signature("other")
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _current_record(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=input_signature,
        )
    )
    check = _check_record(candidate, input_signature=input_signature)
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )
    summary = vpx.plan_to_json(plan)
    context = replay_context_for_plan(plan)
    other_materializer = vpx.CallableMaterializer(
        "tests.materialize_candidate",
        "1",
        {},
        {"callback": "tests.materialize_candidate_impl.other"},
        materialize_candidate_impl,
    )

    assert vpx.plan_to_json(plan) != vpx.plan_to_json(
        dataclasses.replace(
            plan,
            materializers={"family": other_materializer},
        )
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                input_signature=other_signature,
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                family_input_signatures={"family": other_signature},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                selection_policy=vpx.SelectionPolicy(near_fastest_multiplier=1.01),
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                materializer_identities={"family": other_materializer.identity()},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                target_identity={"target": "other", "environment": {}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                runtime_identities={"family": {"runtime": "other"}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                adapter_identities={
                    "family": {"adapter_id": "other", "adapter_version": "1"}
                },
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": other_materializer},
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(),
            materializers={"family": materialize_candidate},
        )

    stale_candidate_row = dict(candidate_records_for_plan(plan)[0])
    stale_candidate_row["candidate_settings"] = {"scale": 2.0}

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(stale_candidate_row,),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_recomputes_family_selection() -> None:
    input_signature = _input_signature("recompute")
    memory_signature = _input_signature("recompute-memory")
    fast = vpx.Candidate(
        "family",
        "fast",
        {"scale": 1.0},
        admission_status="passed",
    )
    slow = vpx.Candidate(
        "family",
        "slow",
        {"scale": 2.0},
        admission_status="passed",
    )
    fast_record = _current_record(
        _record(
            fast,
            elapsed=(1.0,),
            reserved=(10.0,),
            input_signature=input_signature,
        )
    )
    slow_record = _current_record(
        _record(
            slow,
            elapsed=(2.0,),
            reserved=(1.0,),
            input_signature=input_signature,
        )
    )
    fast_check = _check_record(fast, input_signature=input_signature)
    slow_check = _check_record(slow, input_signature=input_signature)
    plan = vpx.Plan(
        selected={"family": slow},
        records={"family": slow_record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(fast_record, slow_record),
        check_records=(fast_check, slow_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError, match=r"selection|selected"):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(fast_record, slow_record),
            check_records=(fast_check, slow_check),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    low_memory = vpx.Candidate(
        "family",
        "low-memory",
        {"scale": 1.0},
        admission_status="passed",
    )
    high_memory = vpx.Candidate(
        "family",
        "high-memory",
        {"scale": 2.0},
        admission_status="passed",
    )
    low_memory_record = _current_record(
        _record(
            low_memory,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=memory_signature,
        )
    )
    high_memory_record = _current_record(
        _record(
            high_memory,
            elapsed=(1.0,),
            reserved=(10.0,),
            input_signature=memory_signature,
        )
    )
    low_memory_check = _check_record(
        low_memory,
        input_signature=memory_signature,
    )
    high_memory_check = _check_record(
        high_memory,
        input_signature=memory_signature,
    )
    memory_plan = vpx.Plan(
        selected={"family": high_memory},
        records={"family": high_memory_record},
        input_signature=memory_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(low_memory_record, high_memory_record),
        check_records=(low_memory_check, high_memory_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError, match=r"selection|selected"):
        vpx.plan_from_json(
            vpx.plan_to_json(memory_plan),
            replay_context=replay_context_for_plan(memory_plan),
            full_size_records=(low_memory_record, high_memory_record),
            check_records=(low_memory_check, high_memory_check),
            candidate_records=candidate_records_for_plan(memory_plan),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_rejects_missing_full_size_agreement() -> None:
    input_signature = _input_signature("full-size-gate-replay")
    candidate = vpx.Candidate(
        "family",
        "flash",
        {
            "attention.frontend": "pytorch_sdpa_direct",
            "attention.sdpa_kernel": "flash_attention",
        },
        admission_status="passed",
    )
    record = _current_record(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=input_signature,
        )
    )
    check = _check_record(candidate, input_signature=input_signature)
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.VPTuneError, match=r"accepted rows|did not pass"):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )


def test_selected_plan_validation_writes_failed_record(tmp_path: Path) -> None:
    input_signature = _input_signature("validation")
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    check = _check_record(candidate, input_signature=input_signature)
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        validation_order=("family",),
        **_identity_kwargs(),
    )

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert record.candidate_id == "row"
        assert torch.equal(context.selected(), torch.tensor([1.0]))
        message = "selected row failed validation"
        raise vp.ReferenceFailedError(message)

    with pytest.raises(vp.ReferenceFailedError):
        vp.validate_plan(plan, {"family": validator}, run_dir=tmp_path)

    failed = read_record(
        tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
    )
    summary = read_record(tmp_path / "summaries" / "selected_plan_validation.json")

    assert failed["status"] == "failed"
    assert failed["error_type"] == "ReferenceFailed"
    assert summary["status"] == "failed"
    failed_record = vpx.check_record_from_json(failed)

    assert summary["records"] == [failed_record.row_key()]

    assert vpx.selected_plan_validation_summary_current(
        summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_summary=summary,
            validation_records=(failed_record,),
        )

    stale_summary = dict(summary)
    stale_summary["records"] = []

    assert not vpx.selected_plan_validation_summary_current(
        stale_summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_summary=stale_summary,
            validation_records=(failed_record,),
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_records=(failed_record,),
        )

    with pytest.raises(vp.MaterializationError):
        vp.validate_plan(plan, {})


def test_selected_plan_validation_writes_runtime_failure_record(
    tmp_path: Path,
) -> None:
    input_signature = _input_signature("validation-runtime")
    candidate = vpx.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature=input_signature,
    )
    plan = vpx.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature=input_signature,
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        materializers={"family": materialize_candidate},
    )

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert record.candidate_id == "row"
        assert torch.equal(context.selected(), torch.tensor([1.0]))
        message = "validation runtime failed"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError):
        vp.validate_plan(plan, {"family": validator}, run_dir=tmp_path)

    failed = read_record(
        tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
    )
    summary = read_record(tmp_path / "summaries" / "selected_plan_validation.json")

    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"
    assert summary["status"] == "failed"


def test_operator_constructors_reject_unknown_semantic_values() -> None:
    with pytest.raises(vp.MaterializationError, match="aggregation"):
        ops.gradient("grad", "loss", aggregation="summ")

    with pytest.raises(vp.MaterializationError, match="distribution"):
        ops.fisher_vp(
            "fisher",
            "scores",
            aggregation="mean",
            distribution="explicit_scores_gradient",
            label_policy="explicit_scores",
            sample_space="terms",
            score_reduction="none",
            denominator="num_examples",
        )

    with pytest.raises(vp.MaterializationError, match="label_policy"):
        ops.fisher_vp(
            "fisher",
            "scores",
            aggregation="mean",
            distribution="explicit_score_gradients",
            label_policy="sampled_labels",
            sample_space="terms",
            score_reduction="none",
            denominator="num_examples",
        )

    with pytest.raises(vp.MaterializationError, match="sample_space"):
        ops.fisher_vp(
            "fisher",
            "scores",
            aggregation="mean",
            distribution="explicit_score_gradients",
            label_policy="explicit_scores",
            sample_space="classes",
            score_reduction="none",
            denominator="num_examples",
        )

    with pytest.raises(vp.MaterializationError, match="score_reduction"):
        ops.sampled_fisher_vp(
            "sampled",
            "scores",
            aggregation="mean",
            distribution="explicit_score_gradients",
            label_policy="sampled_labels",
            sample_count=1,
            sample_source="fixed_seed_and_count",
            sampling_bound=valid_sampling_bound(),
            score_reduction="mean",
            denominator="num_examples",
        )

    with pytest.raises(vp.MaterializationError, match="denominator"):
        ops.empirical_fisher_vp(
            "empirical",
            "loss",
            aggregation="mean",
            example_loss_reduction="per_example",
            denominator="batch",
        )

    with pytest.raises(vp.MaterializationError, match="example_loss_reduction"):
        ops.empirical_fisher_vp(
            "empirical",
            "loss",
            aggregation="mean",
            example_loss_reduction="mean_loss",
            denominator="num_examples",
        )


def test_tune_run_preflight_errors_do_not_write_summary(tmp_path: Path) -> None:
    target = cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss_a", aggregation="sum")
    operator_b = ops.gradient("b", "loss_b", aggregation="sum")

    def make_problem(operator: vpx.OperatorSpec) -> vpx.Problem:
        candidate = passed_candidate(operator.family, "row", {})

        return vpx.Problem(
            model=model,
            params=vpx.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=runtime_config(
                (candidate,),
                operation_factory_never_runs,
                reference_check_never_runs,
                materialize_candidate,
                None,
                {"generator": operator.family},
            ),
        )

    cases = (
        (
            tmp_path / "missing-problems",
            vpx.TuningRun(
                target=target,
                families=(vpx.Family("a", operator_a),),
                run_id="missing-problems",
            ),
            "run adapter is required",
        ),
        (
            tmp_path / "duplicate-problems",
            vpx.TuningRun(
                target=target,
                families=(vpx.Family("a", operator_a),),
                problems=(make_problem(operator_a), make_problem(operator_a)),
                run_id="duplicate-problems",
            ),
            "problem operator families must be unique",
        ),
        (
            tmp_path / "family-mismatch",
            vpx.TuningRun(
                target=target,
                families=(vpx.Family("b", operator_b),),
                problems=(make_problem(operator_a),),
                run_id="family-mismatch",
            ),
            "run families must match",
        ),
    )

    for run_dir, run, message in cases:
        with pytest.raises(vp.MaterializationError, match=message):
            vp.tune_run(
                run,
                run_dir=run_dir,
                memory_backend=CPUMemoryBackend(),
                clock=SequenceClock(()),
            )

        assert not (run_dir / "summaries" / "tuning.json").exists()


def test_tune_run_uses_family_dag_order(tmp_path: Path) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    target = one_call_cpu_target()
    operator_a = ops.gradient("a", "loss_a", aggregation="sum")
    operator_b = ops.hvp("b", "loss_b", aggregation="sum")

    run = vpx.TuningRun(
        target=target,
        families=(
            vpx.Family("b", operator_b, dependencies=("a",)),
            vpx.Family("a", operator_a),
        ),
        problems=(
            recorded_tuning_problem(
                model=model,
                target=target,
                name="b",
                operator=operator_b,
                candidates=passed_candidates("b", (("b:row", {"axis": "b"}),)),
                calls=calls,
                record_call=lambda candidate: candidate.family,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="a",
                operator=operator_a,
                candidates=passed_candidates("a", (("a:row", {"axis": "a"}),)),
                calls=calls,
                record_call=lambda candidate: candidate.family,
            ),
        ),
        run_id="dag",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 2.0)),
    )

    assert calls == ["a", "b"]
    assert tuple(plan.selected) == ("a", "b")
    assert plan.validation_order == ("a", "b")
    assert plan.dependencies_by_family == {"a": (), "b": ("a",)}
    assert len(plan.full_size_records) == 2
    dependency_identity = plan.selected["b"].dependency_identities["a"]
    expected_dependency_identity = {
        "family": "a",
        "candidate_id": "a:row",
        "candidate_settings": dict(plan.selected["a"].settings),
        "full_size_row": plan.records["a"].row_key(),
        "materializer_identity": materialize_candidate.identity(),
    }

    assert dependency_identity == expected_dependency_identity
    assert plan.records["b"].dependency_identities["a"] == expected_dependency_identity
    assert plan.selected_dependency_identities() == {
        "a": {},
        "b": {"a": expected_dependency_identity},
    }

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
    )
    loaded_run = vp.load_tuned_run(
        tmp_path,
        run,
        memory_backend=CPUMemoryBackend(),
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)
    assert vpx.plan_record_current(vpx.plan_to_json(loaded_run), plan)

    saved_paths = tuple(
        sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json"))
    )
    second_plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock(()),
    )

    assert calls == ["a", "b"]
    assert vpx.plan_record_current(vpx.plan_to_json(second_plan), plan)
    assert (
        tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json")))
        == saved_paths
    )

    stale_dependency_identity = dict(expected_dependency_identity)
    stale_dependency_identity["full_size_row"] = {
        **dict(expected_dependency_identity["full_size_row"]),
        "candidate_id": "stale",
    }
    stale_child = dataclasses.replace(
        plan.selected["b"],
        dependency_identities={"a": stale_dependency_identity},
    )
    stale_plan = dataclasses.replace(
        plan,
        selected={"a": plan.selected["a"], "b": stale_child},
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(stale_plan),
            replay_context=replay_context_for_plan(stale_plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(stale_plan),
            materializers=plan.materializers,
        )

    validation_calls = []

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        validation_calls.append(candidate.family)
        assert record.family == candidate.family
        assert torch.equal(context.selected(), torch.tensor([1.0]))

        if candidate.family == "b":
            assert tuple(context.dependencies) == ("a",)
            assert torch.equal(context.dependencies["a"](), torch.tensor([1.0]))

        return reference_passed()

    plan_without_candidate_dependency = dataclasses.replace(
        plan,
        selected={
            "a": plan.selected["a"],
            "b": dataclasses.replace(
                plan.selected["b"],
                dependency_identities={},
            ),
        },
    )

    with pytest.raises(vp.MaterializationError):
        vp.validate_plan(
            plan_without_candidate_dependency, {"a": validator, "b": validator}
        )

    validators_by_family = {"a": validator, "b": validator}
    validation_plan = dataclasses.replace(
        plan,
        validation_required=True,
        validator_identities={
            "a": {"validator": "dag-validator"},
            "b": {"validator": "dag-validator"},
        },
    )
    validation_records = tuple(
        vpx.check_record_from_json(vpx.check_record_to_json(record))
        for record in vp.validate_plan(validation_plan, validators_by_family)
    )
    validated_plan = dataclasses.replace(
        validation_plan,
        validation_records=validation_records,
    )
    validation_summary = vpx.selected_plan_validation_summary_record(
        validation_plan,
        validation_records,
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=validation_records,
        )

    replayed_with_validation = vpx.plan_from_json(
        vpx.plan_to_json(validated_plan),
        replay_context=replay_context_for_plan(
            validated_plan,
            validation_required=True,
        ),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(validated_plan),
        materializers=plan.materializers,
        validation_summary=validation_summary,
        validation_records=validation_records,
    )

    assert replayed_with_validation.validation_required
    assert vpx.plan_record_current(
        vpx.plan_to_json(replayed_with_validation),
        validated_plan,
    )

    changed_validation_record = dataclasses.replace(
        validation_records[0],
        candidate_settings={"axis": "changed"},
    )
    changed_validation_records = (
        changed_validation_record,
        *validation_records[1:],
    )

    assert not vpx.selected_plan_validation_summary_current(
        validation_summary,
        validation_plan,
        changed_validation_records,
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(validated_plan),
            replay_context=replay_context_for_plan(
                validated_plan,
                validation_required=True,
            ),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(validated_plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=changed_validation_records,
        )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(validated_plan),
            replay_context=replay_context_for_plan(
                validated_plan,
                validation_required=True,
            ),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(validated_plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=tuple(reversed(validation_records)),
        )

    assert validation_calls == ["a", "b"]


def test_tune_run_executes_declared_selected_plan_validators(
    tmp_path: Path,
) -> None:
    calls = []
    validation_calls = []
    model = torch.nn.Linear(1, 1)
    target = one_call_cpu_target()
    operator = ops.gradient("family", "loss", aggregation="sum")
    candidate = passed_candidate(
        "family",
        "row",
        {"axis": "value"},
    )

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        validation_calls.append(candidate.candidate_id)
        assert record.candidate_id == candidate.candidate_id
        assert torch.equal(context.selected(), torch.tensor([1.0]))

        return vpx.ReferenceResult(
            "selected_plan_validation",
            {"max_abs_diff": 1e-6},
            {"max_abs_diff": 0.0},
        )

    problem = recorded_tuning_problem(
        model=model,
        target=target,
        name="family",
        operator=operator,
        candidates=(candidate,),
        calls=calls,
        generator="validation-run",
    )
    run = vpx.TuningRun(
        target=target,
        families=(vpx.Family("family", operator),),
        problems=(problem,),
        validators={"family": validator},
        validator_identities={"family": {"validator_id": "tests.validator.v1"}},
        run_id="validation-run",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    validation_record = vpx.check_record_from_json(
        read_record(
            tmp_path / "references" / "family" / "row" / "selected_plan_validation.json"
        )
    )
    validation_summary = read_record(
        tmp_path / "summaries" / "selected_plan_validation.json"
    )
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan, validation_required=True),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
        validation_summary=validation_summary,
        validation_records=(validation_record,),
    )
    stale_validator_context = dataclasses.replace(
        replay_context_for_plan(plan, validation_required=True),
        validator_identities={"family": {"validator_id": "tests.validator.v2"}},
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=stale_validator_context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=(validation_record,),
        )

    assert calls == ["row"]
    assert validation_calls == ["row"]
    assert plan.validation_required
    assert to_json_value(
        tuple(record.row_key() for record in plan.validation_records)
    ) == (to_json_value((validation_record.row_key(),)))
    assert to_json_value(
        tuple(record.row_key() for record in replayed.validation_records)
    ) == to_json_value((validation_record.row_key(),))
    assert plan.validator_identities == {
        "family": {"validator_id": "tests.validator.v1"}
    }
    assert validation_summary["status"] == "passed"

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
        )

    forged_record = dataclasses.replace(
        validation_record,
        name="tree_close",
    )
    forged_summary = vpx.selected_plan_validation_summary_record(
        plan,
        (forged_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=forged_summary,
            validation_records=(forged_record,),
        )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

    bad_dir = tmp_path / "bad-validator"
    bad_run = dataclasses.replace(
        run,
        validators={"wrong": validator},
        validator_identities={"wrong": {"validator_id": "tests.bad"}},
    )

    with pytest.raises(vp.MaterializationError):
        vp.tune_run(
            bad_run,
            run_dir=bad_dir,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    assert not (bad_dir / "summaries" / "tuning.json").exists()


def test_tune_run_writes_selected_summary_before_validator_failure(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    target = one_call_cpu_target()
    operator = ops.gradient("family", "loss", aggregation="sum")
    candidate = passed_candidate("family", "row", {})

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.candidate_id == record.candidate_id
        assert context.family == "family"

        message = "validation failed"
        raise RuntimeError(message)

    problem = recorded_tuning_problem(
        model=model,
        target=target,
        name="family",
        operator=operator,
        candidates=(candidate,),
        calls=[],
        generator="validation-failure-run",
    )
    run = vpx.TuningRun(
        target=target,
        families=(vpx.Family("family", operator),),
        problems=(problem,),
        validators={"family": validator},
        validator_identities={"family": {"validator_id": "tests.validator.fail"}},
        run_id="validation-failure-run",
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        vp.tune_run(
            run,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock((0.0, 1.0)),
        )

    tuning_summary = read_record(tmp_path / "summaries" / "tuning.json")
    validation_summary = read_record(
        tmp_path / "summaries" / "selected_plan_validation.json"
    )

    assert tuning_summary["validation_required"] is True
    assert tuning_summary["validation_records"] == []
    assert validation_summary["status"] == "failed"


def test_tune_run_selects_complete_dtype_cohort(tmp_path: Path) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss", aggregation="sum")
    operator_b = ops.gradient("b", "loss", aggregation="sum")
    calls = []

    run = vpx.TuningRun(
        target=target,
        families=(vpx.Family("a", operator_a), vpx.Family("b", operator_b)),
        problems=(
            recorded_tuning_problem(
                model=model,
                target=target,
                name="a",
                operator=operator_a,
                candidates=passed_candidates(
                    "a",
                    (
                        ("a-float16", {"dtype.model_compute": "fp16"}),
                        ("a-float32", {"dtype.model_compute": "fp32"}),
                    ),
                ),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="b",
                operator=operator_b,
                candidates=passed_candidates(
                    "b",
                    (
                        ("b-float16", {"dtype.model_compute": "fp16"}),
                        ("b-float32", {"dtype.model_compute": "fp32"}),
                    ),
                ),
                calls=calls,
            ),
        ),
        cohort_constraints=(
            vpx.CohortConstraint(
                name="dtype",
                settings_keys=("dtype.model_compute",),
                assignments=(
                    {"dtype.model_compute": "fp16"},
                    {"dtype.model_compute": "fp32"},
                ),
            ),
        ),
        run_id="dtype-cohort",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 101.0, 101.0, 111.0, 111.0, 121.0)),
    )

    assert calls == ["a-float16", "b-float16", "a-float32", "b-float32"]
    assert plan.selected["a"].candidate_id == "a-float32"
    assert plan.selected["b"].candidate_id == "b-float32"
    assert plan.cohort_assignment is not None
    assert plan.cohort_assignment.values == {"dtype.model_compute": "fp32"}


def test_tune_run_uses_generic_multi_key_cohort_constraint(tmp_path: Path) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss", aggregation="sum")
    operator_b = ops.gradient("b", "loss", aggregation="sum")
    operator_c = ops.gradient("c", "loss", aggregation="sum")
    calls = []

    run = vpx.TuningRun(
        target=target,
        families=(
            vpx.Family("a", operator_a),
            vpx.Family("b", operator_b, dependencies=("a",)),
            vpx.Family("c", operator_c),
        ),
        problems=(
            recorded_tuning_problem(
                model=model,
                target=target,
                name="a",
                operator=operator_a,
                candidates=passed_candidates(
                    "a",
                    (
                        ("a-first", {"backend": "first", "chunk": 1}),
                        ("a-second", {"backend": "second", "chunk": 2}),
                    ),
                ),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="b",
                operator=operator_b,
                candidates=passed_candidates(
                    "b",
                    (
                        ("b-first", {"backend": "first", "chunk": 1}),
                        ("b-second", {"backend": "second", "chunk": 2}),
                    ),
                ),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="c",
                operator=operator_c,
                candidates=passed_candidates("c", (("c-row", {}),)),
                calls=calls,
            ),
        ),
        cohort_constraints=(
            vpx.CohortConstraint(
                name="backend_chunk",
                settings_keys=("backend", "chunk"),
                assignments=(
                    {"backend": "first", "chunk": 1},
                    {"backend": "second", "chunk": 2},
                ),
                families=("a", "b"),
            ),
        ),
        run_id="generic-cohort",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            1.0,
            1.0,
            101.0,
            101.0,
            102.0,
            102.0,
            112.0,
            112.0,
            122.0,
            122.0,
            123.0,
        )),
    )

    assert calls == ["a-first", "b-first", "c-row", "a-second", "b-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert plan.selected["c"].candidate_id == "c-row"
    assert plan.cohort_assignment is not None
    assert plan.cohort_assignment.values == {"backend": "second", "chunk": 2}
    assert plan.selected["b"].dependency_identities["a"]["candidate_id"] == "a-second"

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    candidate_rows = candidate_records_for_plan(plan)
    replayed = vpx.plan_from_json(
        vpx.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_rows,
        materializers=plan.materializers,
    )

    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)

    changed_cohort_record = dataclasses.replace(
        saved_full_size[0],
        cohort_assignment={"assignment_id": "stale"},
    )

    with pytest.raises(vp.VPTuneError):
        vpx.plan_from_json(
            vpx.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(changed_cohort_record, *saved_full_size[1:]),
            check_records=saved_checks,
            candidate_records=candidate_rows,
            materializers=plan.materializers,
        )

    candidates_by_key = {
        (
            candidate.family,
            candidate.candidate_id,
            canonical_json(candidate.settings),
        ): candidate
        for candidate in (
            vpx.candidate_record_from_json(candidate_row)
            for candidate_row in candidate_rows
        )
    }
    first_records = {
        "a": plan.full_size_records[0],
        "b": plan.full_size_records[1],
        "c": plan.full_size_records[2],
    }
    first_selected = {
        family: candidates_by_key[
            record.family,
            record.candidate_id,
            canonical_json(record.candidate_settings),
        ]
        for family, record in first_records.items()
    }
    first_assignment_record = first_selected["a"].cohort_assignment
    stale_plan = dataclasses.replace(
        plan,
        selected=first_selected,
        records=first_records,
        cohort_assignment=vpx.CohortAssignment(
            assignment_id=str(first_assignment_record["assignment_id"]),
            values=dict(first_assignment_record["values"]),
            constraints=tuple(
                str(name) for name in first_assignment_record["constraints"]
            ),
            covered_families=tuple(
                str(family) for family in first_assignment_record["covered_families"]
            ),
        ),
    )

    with pytest.raises(vp.StaleRecordError):
        vpx.plan_from_json(
            vpx.plan_to_json(stale_plan),
            replay_context=replay_context_for_plan(stale_plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_rows,
            materializers=plan.materializers,
        )


def test_tune_run_cohort_subset_handles_cross_boundary_dependencies(
    tmp_path: Path,
) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss", aggregation="sum")
    operator_b = ops.gradient("b", "loss", aggregation="sum")
    operator_c = ops.gradient("c", "loss", aggregation="sum")
    calls = []

    run = vpx.TuningRun(
        target=target,
        families=(
            vpx.Family("a", operator_a),
            vpx.Family("b", operator_b, dependencies=("a",)),
            vpx.Family("c", operator_c, dependencies=("b",)),
        ),
        problems=(
            recorded_tuning_problem(
                model=model,
                target=target,
                name="a",
                operator=operator_a,
                candidates=passed_candidates(
                    "a",
                    (
                        ("a-first", {"backend": "first"}),
                        ("a-second", {"backend": "second"}),
                    ),
                ),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="b",
                operator=operator_b,
                candidates=passed_candidates("b", (("b-row", {}),)),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="c",
                operator=operator_c,
                candidates=passed_candidates(
                    "c",
                    (
                        ("c-first", {"backend": "first"}),
                        ("c-second", {"backend": "second"}),
                    ),
                ),
                calls=calls,
            ),
        ),
        cohort_constraints=(
            vpx.CohortConstraint(
                name="backend",
                settings_keys=("backend",),
                assignments=({"backend": "first"}, {"backend": "second"}),
                families=("a", "c"),
            ),
        ),
        run_id="cohort-cross-boundary-deps",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            10.0,
            10.0,
            11.0,
            11.0,
            21.0,
            21.0,
            22.0,
            22.0,
            23.0,
            23.0,
            24.0,
        )),
    )
    replayed = vp.load_tuned_run(
        tmp_path,
        run,
        memory_backend=CPUMemoryBackend(),
    )

    assert calls == ["a-first", "b-row", "c-first", "a-second", "b-row", "c-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-row"
    assert plan.selected["c"].candidate_id == "c-second"
    assert plan.selected["b"].dependency_identities["a"]["candidate_id"] == "a-second"
    assert plan.selected["c"].dependency_identities["b"]["candidate_id"] == "b-row"
    assert vpx.plan_record_current(vpx.plan_to_json(replayed), plan)


def test_tune_run_writes_prerequisite_failed_descendants(tmp_path: Path) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss", aggregation="sum")
    operator_b = ops.gradient("b", "loss", aggregation="sum")
    operator_c = ops.gradient("c", "loss", aggregation="sum")
    calls = []

    run = vpx.TuningRun(
        target=target,
        families=(
            vpx.Family("a", operator_a),
            vpx.Family("b", operator_b, dependencies=("a",)),
            vpx.Family("c", operator_c),
        ),
        problems=(
            recorded_tuning_problem(
                model=model,
                target=target,
                name="a",
                operator=operator_a,
                candidates=passed_candidates(
                    "a",
                    (
                        ("a-first", {"backend": "first"}),
                        ("a-second", {"backend": "second"}),
                    ),
                ),
                calls=calls,
                failing_reference_candidate="a-first",
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="b",
                operator=operator_b,
                candidates=passed_candidates(
                    "b",
                    (
                        ("b-first", {"backend": "first"}),
                        ("b-second", {"backend": "second"}),
                    ),
                ),
                calls=calls,
            ),
            recorded_tuning_problem(
                model=model,
                target=target,
                name="c",
                operator=operator_c,
                candidates=passed_candidates(
                    "c",
                    (
                        ("c-first", {"backend": "first"}),
                        ("c-second", {"backend": "second"}),
                    ),
                ),
                calls=calls,
            ),
        ),
        cohort_constraints=(
            vpx.CohortConstraint(
                name="backend",
                settings_keys=("backend",),
                assignments=({"backend": "first"}, {"backend": "second"}),
            ),
        ),
        run_id="blocked-descendant",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0)),
    )

    blocked = tuple(
        record
        for record in plan.full_size_records
        if record.error_type == "PrerequisiteFailed"
    )
    failed_references = tuple(
        record
        for record in plan.full_size_records
        if record.error_type == "RuntimeError"
    )

    assert calls == ["c-first", "a-second", "b-second", "c-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert plan.selected["c"].candidate_id == "c-second"
    assert tuple(record.candidate_id for record in failed_references) == ("a-first",)
    assert tuple(record.candidate_id for record in blocked) == ("b-first",)


def test_tune_run_propagates_candidate_validation_errors_inside_cohort(
    tmp_path: Path,
) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator = ops.gradient("family", "loss", aggregation="sum")
    candidates = passed_candidates(
        "family",
        (
            ("duplicate", {"backend": "first"}),
            ("duplicate", {"backend": "first"}),
            ("valid", {"backend": "second"}),
        ),
    )

    problem = recorded_tuning_problem(
        model=model,
        target=target,
        name="family",
        operator=operator,
        candidates=candidates,
        calls=[],
        generator="validation-error",
    )
    run = vpx.TuningRun(
        target=target,
        families=(vpx.Family("family", operator),),
        problems=(problem,),
        cohort_constraints=(
            vpx.CohortConstraint(
                name="backend",
                settings_keys=("backend",),
                assignments=({"backend": "first"}, {"backend": "second"}),
            ),
        ),
        run_id="cohort-validation-error",
    )

    with pytest.raises(vp.MaterializationError, match="candidate ids must be unique"):
        vp.tune_run(
            run,
            run_dir=tmp_path,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock((0.0, 1.0)),
        )


PACKAGE_LAYER_BY_OWNER = {
    "errors": 0,
    "core": 1,
    "axes": 2,
    "engine": 3,
    "tuning": 4,
    "adapters": 4,
    "public": 5,
    "ext": 5,
    "__init__": 5,
}


def test_package_layers_have_one_directional_imports() -> None:
    src_root = repo_root() / "src" / "vptune"

    for path in sorted(src_root.rglob("*.py")):
        owner = path.relative_to(src_root).parts[0].removesuffix(".py")
        source_layer = PACKAGE_LAYER_BY_OWNER[owner]
        tree = ast.parse(path.read_text())

        for node in ast.walk(tree):
            targets = []

            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "vptune":
                    targets = [alias.name for alias in node.names]
                elif node.module.startswith("vptune."):
                    targets = [node.module.split(".")[1]]
            elif isinstance(node, ast.Import):
                targets = [
                    alias.name.split(".")[1]
                    for alias in node.names
                    if alias.name.startswith("vptune.")
                ]

            for target in targets:
                target_layer = PACKAGE_LAYER_BY_OWNER[target]
                assert target_layer <= source_layer, (
                    f"{owner} (layer {source_layer}) imports "
                    f"{target} (layer {target_layer}) in {path}"
                )


def test_matrix_free_inverse_metric_rows_admit_only_conjugate_gradient() -> None:
    runtime = runtime_config(
        passed_candidates(
            "inverse_metric",
            (
                (
                    "cg",
                    {"inverse_metric.solve_path": "conjugate_gradient"},
                ),
                (
                    "cholesky",
                    {"inverse_metric.solve_path": "cholesky_solve"},
                ),
                (
                    "eigh",
                    {"inverse_metric.solve_path": "eigh_solve"},
                ),
                (
                    "svd",
                    {"inverse_metric.solve_path": "svd_solve"},
                ),
            ),
        ),
        constant_operation_factory,
        passing_reference_check,
        materialize_candidate,
        None,
        {
            "runtime": "standard",
            "operator": {
                "kind": "inverse_metric",
                "semantics": {
                    "representation": {"kind": "matrix_free"},
                },
            },
        },
    )
    rows = {
        candidate.candidate_id: candidate
        for candidate in run_module._candidate_rows(runtime)
    }

    assert rows["cg"].admission_status == "passed"

    for candidate_id in ("cholesky", "eigh", "svd"):
        assert rows[candidate_id].admission_status == "failed"
        assert rows[candidate_id].admission_error == (
            "metric representation kind is not supported by path: matrix_free"
        )


def test_cohort_constraint_pins_vector_axes_and_rejects_disagreement(
    tmp_path: Path,
) -> None:
    target = one_call_cpu_target()
    model = torch.nn.Linear(1, 1)
    operator_a = ops.gradient("a", "loss", aggregation="sum")
    operator_b = ops.gradient("b", "loss", aggregation="sum")
    constraint = vpx.CohortConstraint(
        name="vector_axes",
        settings_keys=("layout.vector", "dtype.vector"),
        assignments=(
            {"layout.vector": "flat_contiguous", "dtype.vector": "fp32"},
            {"layout.vector": "per_layer_flat", "dtype.vector": "bf16"},
        ),
    )

    def problem_for(
        name: str,
        operator: vpx.OperatorSpec,
        rows: tuple[tuple[str, Mapping[str, Any]], ...],
        calls: list[str],
    ) -> vpx.Problem:
        return recorded_tuning_problem(
            model=model,
            target=target,
            name=name,
            operator=operator,
            candidates=passed_candidates(name, rows),
            calls=calls,
        )

    matching_rows = (
        ("flat", {"layout.vector": "flat_contiguous", "dtype.vector": "fp32"}),
        ("layer", {"layout.vector": "per_layer_flat", "dtype.vector": "bf16"}),
    )
    calls = []
    plan = vp.tune_run(
        vpx.TuningRun(
            target=target,
            families=(vpx.Family("a", operator_a), vpx.Family("b", operator_b)),
            problems=(
                problem_for("a", operator_a, matching_rows, calls),
                problem_for("b", operator_b, matching_rows, calls),
            ),
            cohort_constraints=(constraint,),
            run_id="vector-cohort",
        ),
        run_dir=tmp_path / "agree",
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 101.0, 101.0, 111.0, 111.0, 121.0)),
    )

    assert plan.cohort_assignment is not None
    pinned = plan.cohort_assignment.values

    for family in ("a", "b"):
        selected = plan.selected[family]

        for key in ("layout.vector", "dtype.vector"):
            assert selected.settings[key] == pinned[key]

    with pytest.raises(vp.NoPassedCandidateError):
        vp.tune_run(
            vpx.TuningRun(
                target=target,
                families=(vpx.Family("a", operator_a), vpx.Family("b", operator_b)),
                problems=(
                    problem_for("a", operator_a, matching_rows, []),
                    problem_for(
                        "b",
                        operator_b,
                        (
                            (
                                "mixed",
                                {
                                    "layout.vector": "flat_contiguous",
                                    "dtype.vector": "bf16",
                                },
                            ),
                        ),
                        [],
                    ),
                ),
                cohort_constraints=(constraint,),
                run_id="vector-cohort-disagree",
            ),
            run_dir=tmp_path / "disagree",
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock((0.0, 1.0, 1.0, 101.0, 101.0, 111.0, 111.0, 121.0)),
        )


def test_tune_exhaustive_strategy_measures_every_admitted_row(
    tmp_path: Path,
) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidates = passed_changed_candidates(
        "family",
        (
            ("base", {}, ()),
            (
                "hvp-path",
                {"hvp.path": "reverse_over_reverse"},
                ("hvp.path",),
            ),
            (
                "gradient-graph",
                {"gradient.graph_schedule": "build_once"},
                ("gradient.graph_schedule",),
            ),
            ("dtype", {"dtype.model_compute": "fp32"}, ("dtype.model_compute",)),
        ),
    )
    target = dataclasses.replace(
        cpu_target(
            vpx.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_measured_calls=1,
            )
        ),
        search_policy=vpx.SearchPolicy(strategy="exhaustive"),
    )

    def operation_factory(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        del batch, vector
        calls.append(("operation", candidate.candidate_id))

        return vpx.constant_operation(torch.tensor([1.0]))

    def reference_check(
        candidate: vpx.Candidate,
        batch: vpx.Batch,
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        del batch, vector
        calls.append(("reference", candidate.candidate_id))

        return reference_passed()

    problem = vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=TwoProbeData(),
        operator=ops.gradient("family", "loss", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=target,
        runtime=runtime_config(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"runtime": "test.search-exhaustive"},
        ),
    )
    plan = tune_problem(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
    )

    measured = tuple(record.candidate_id for record in plan.full_size_records)

    assert sorted(measured) == ["base", "dtype", "gradient-graph", "hvp-path"]
    assert sorted({call[1] for call in calls}) == [
        "base",
        "dtype",
        "gradient-graph",
        "hvp-path",
    ]


def test_acceptance_mapped_tests_contain_behavioral_assertions() -> None:
    nodes = collected_test_function_nodes()
    weak = {
        name
        for names in ACCEPTANCE_TEST_COVERAGE.values()
        for name in names
        if not has_behavioral_assertion(nodes[name])
    }

    assert sorted(weak) == []


def test_manifest_rejects_declared_contradictory_rows() -> None:
    registry = vpx.standard_axis_registry()

    packed = registry.admit(
        vpx.Candidate(
            "gradient",
            "packed-dense",
            {
                "schedule.per_token": "packed",
                "input.batch_layout": "dense_padded",
            },
        )
    )

    assert packed.admission_status == "failed"
    assert packed.admission_error == (
        "schedule.per_token=packed requires input.batch_layout "
        "packed_with_inverse_permutation or variable_length"
    )

    for alias in ("reduce-overhead", "max-autotune-no-cudagraphs"):
        aliased = registry.admit(
            vpx.Candidate("gradient", f"alias-{alias}", {"compile.mode": alias})
        )

        assert aliased.admission_status == "failed"
        assert aliased.admission_error == (
            "candidate axis value is not allowed: compile.mode"
        )

    admitted, error = vpx.admit_core_attention(
        vpx.Candidate(
            "attention",
            "kernel-on-eager",
            {
                "attention.frontend": "patched_eager",
                "attention.sdpa_kernel": "math",
                "attention.partition": "full",
                "attention.padding": "dense_padded",
            },
        )
    )

    assert not admitted
    assert error == "attention.sdpa_kernel applies only to pytorch_sdpa_direct"
