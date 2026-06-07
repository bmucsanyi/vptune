"""Runtime builders for package-owned operator anchors."""

import contextlib
import dataclasses
import importlib
import inspect
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from itertools import starmap
from typing import Any, TypeGuard

import torch
from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune.admission import (
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_call_core_settings,
    admit_checkpoint,
    admit_forward_ad,
    admit_functional_call,
    admit_torch_func,
)
from vptune.anchors import (
    dense_metric_inverse_multiply,
    dense_metric_multiply,
    finite_difference_hvp,
    finite_difference_jvp,
    forward_ad_jvp_anchor,
    gradient_anchor,
    hvp_anchor,
    hvp_jvp_grad_anchor,
    hvp_reverse_over_reverse_anchor,
    jvp_anchor,
    vjp_dot_identity_error,
)
from vptune.candidates import standard_axis_registry
from vptune.checks import (
    STANDARD_THRESHOLDS,
    numeric_error_bound_measurements,
    tree_error_measurements,
    validate_numeric_error_bound,
    validate_thresholds,
)
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    BufferTree,
    CallableMaterializer,
    CallableOperationFactory,
    CallableReferenceCheck,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    DataProvider,
    FullSizeRecord,
    FunctionObjective,
    Materializer,
    ModuleCallSpec,
    ObjectiveContext,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    Problem,
    ReferenceCheck,
    ReferenceChildResult,
    ReferenceResult,
    RuntimeConfig,
    RuntimeOperationFactory,
    RuntimeReferenceCheck,
    ScalarObjective,
    Target,
    VectorProvider,
)
from vptune.errors import AdmissionError, MaterializationError, ReferenceFailedError
from vptune.identities import stable_hash, to_json_value
from vptune.tensor_tree import (
    TensorTree,
    tree_add_foreach,
    tree_add_scalar_foreach,
    tree_dot,
    tree_dot_foreach,
    tree_elementwise_div_foreach,
    tree_elementwise_mul_foreach,
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_map2,
    tree_mul_foreach,
    tree_signature,
)

MATRIX_FREE_RUNTIME_BINDINGS = ContextVar[
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]] | None
](
    "vptune_matrix_free_runtime_bindings",
    default=None,
)

GRADIENT_PATH = "autograd_grad"
GRADIENT_TORCH_FUNC_PATH = "torch_func_grad"
GRADIENT_TORCH_FUNC_VALUE_PATH = "torch_func_grad_and_value"
GRADIENT_BACKWARD_MATERIALIZED_PATH = "backward_materialized_grad"
JVP_PATH = "torch_func_jvp"
JVP_FORWARD_AD_PATH = "forward_ad_jvp"
JVP_LINEARIZE_PATH = "torch_func_linearize"
VJP_PATH = "torch_func_vjp"
VJP_AUTOGRAD_OUTPUTS_PATH = "autograd_grad_outputs"
VJP_BACKWARD_MATERIALIZED_PATH = "backward_materialized_grad"
HVP_REFERENCE_PATH = "reverse_over_reverse"
HVP_FUNCTIONAL_PATH = "functional_hvp"
HVP_JVP_GRAD_PATH = "jvp_grad"
HVP_FORWARD_AD_PATH = "forward_ad_hvp"
HVP_LINEARIZE_GRAD_PATH = "linearize_grad"
VHP_PATH = "vhp"
GGN_DENSE_PATH = "dense_ggn"
GGN_JVP_HESSIAN_VJP_PATH = "jvp_hessian_vjp"
GGN_FORWARD_AD_HESSIAN_VJP_PATH = "forward_ad_hessian_vjp"
GGN_LINEARIZE_HESSIAN_VJP_PATH = "linearize_hessian_vjp"
FISHER_DENSE_PATH = "dense_score_outer"
FISHER_SCORE_GRADIENT_LOOP_PATH = "score_gradient_loop"
FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH = "score_gradient_torch_func"
FISHER_SCORE_GRADIENT_VMAP_PATH = "score_gradient_vmap"
FISHER_BACKWARD_MATERIALIZED_PATH = "score_gradient_backward_materialized"
FISHER_BLOCKWISE_SCORE_MATRIX_PATH = "blockwise_score_matrix"
SAMPLED_FISHER_DENSE_PATH = "dense_sampled_score_outer"
SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH = "sampled_score_gradient_loop"
SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH = "sampled_score_gradient_torch_func"
SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH = "sampled_score_gradient_vmap"
SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH = (
    "sampled_score_gradient_backward_materialized"
)
SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH = "sampled_blockwise_score_matrix"
EMPIRICAL_FISHER_DENSE_PATH = "dense_empirical_fisher"
EMPIRICAL_FISHER_GRADIENT_LOOP_PATH = "per_example_gradient_loop"
EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH = "per_example_torch_func_grad_loop"
EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH = "per_example_backward_materialized"
EMPIRICAL_FISHER_GRADIENT_VMAP_PATH = "per_example_gradient_vmap"
EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH = "blockwise_gradient_matrix"
PER_EXAMPLE_GRADIENT_LOOP_PATH = "stacked_per_example_gradient_loop"
PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH = "stacked_per_example_torch_func_grad_loop"
PER_EXAMPLE_GRADIENT_BACKWARD_PATH = "stacked_per_example_backward_materialized"
PER_EXAMPLE_GRADIENT_VMAP_PATH = "stacked_per_example_gradient_vmap"
METRIC_DENSE_PATH = "dense_metric"
METRIC_FACTORIZED_PATH = "factorized_metric"
METRIC_BLOCKWISE_PATH = "blockwise_metric"
METRIC_STREAMING_PATH = "streaming_metric"
SQRT_METRIC_CLOSED_FORM_PATH = "closed_form_metric_square_root"
SQRT_METRIC_CHOLESKY_PATH = "cholesky_metric_square_root"
SQRT_METRIC_EIGENBASIS_PATH = "eigenbasis_metric_square_root"
SQRT_METRIC_LANCZOS_PATH = "matrix_free_lanczos_metric_square_root"
METRIC_INNER_MULTIPLY_REDUCE_PATH = "metric_inner_multiply_then_reduce"
METRIC_INNER_FACTORED_GRAM_PATH = "metric_inner_factored_gram"
METRIC_INNER_SQRT_REDUCE_PATH = "metric_inner_sqrt_apply_reduce"
METRIC_INNER_VECTOR_COUNT = 2
INVERSE_METRIC_DENSE_PATH = "dense_inverse_metric"
INVERSE_METRIC_CG_PATH = "conjugate_gradient_inverse_metric"
INVERSE_METRIC_CHOLESKY_PATH = "cholesky_inverse_metric"
INVERSE_METRIC_EIGH_PATH = "eigh_inverse_metric"
INVERSE_METRIC_SVD_PATH = "svd_inverse_metric"
INVERSE_METRIC_FACTORIZED_PATH = "factorized_inverse_metric"
INVERSE_METRIC_BLOCKWISE_PATH = "blockwise_inverse_metric"
INVERSE_METRIC_WOODBURY_PATH = "woodbury_low_rank_inverse_metric"
INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH = "inverse_metric_inner_solve_then_reduce"
INVERSE_METRIC_INNER_FACTORED_GRAM_PATH = "inverse_metric_inner_factored_gram"
INVERSE_METRIC_INNER_SQRT_REDUCE_PATH = "inverse_metric_inner_sqrt_apply_reduce"
COMPOSITION_PATH = "sequential_composition"
RUNTIME_DTYPE_SETTINGS = (
    "dtype.parameter_storage",
    "dtype.model_compute",
    "dtype.autodiff_compute",
    "dtype.accumulation",
    "dtype.vector",
    "dtype.intermediate",
    "dtype.metric_factor",
    "dtype.output",
)
METRIC_FACTOR_BATCH_KEYS = (
    "metric_diagonal",
    "low_rank_factors",
    "kfac_factors",
    "ekfac_eigvecs_a",
    "ekfac_eigvecs_g",
    "ekfac_corrected_eigenvalues",
    "ggn_factors",
)
SPEC_PATH_KEYS = {
    "gradient": "gradient.path",
    "jvp": "jvp.path",
    "vjp": "vjp.path",
    "hvp": "hvp.path",
    "ggnvp": "ggn.jvp_path",
    "fisher_vp": "fisher.accumulation",
    "sampled_fisher_vp": "sampled_fisher.accumulation",
    "empirical_fisher_vp": "empirical_fisher.grad_path",
    "per_example_gradient": "per_example_gradient.grad_path",
    "metric": "metric.multiply_path",
    "sqrt_metric": "sqrt_metric.factor_path",
    "inverse_sqrt_metric": "sqrt_metric.factor_path",
    "metric_inner": "metric_inner.reduction_path",
    "inverse_metric": "inverse_metric.solve_path",
    "inverse_metric_inner": "inverse_metric_inner.reduction_path",
    "composition": "composition.execution",
}
SPEC_ADDITIONAL_RUNTIME_SETTINGS = (
    "gradient.value_reuse",
    "gradient.graph_schedule",
    "jvp.linearize_reuse",
    "vjp.closure_reuse",
    "hvp.graph_schedule",
    "hvp.primal_reuse",
    "hvp.gradient_reuse",
    "ggn.loss_hessian_path",
    "ggn.loss_hessian_kernel",
    "ggn.vjp_path",
    "ggn.jvp_reuse",
    "ggn.cotangent_reuse",
    "fisher.expectation_path",
    "fisher.score_grad_path",
    "sampled_fisher.sample_source",
    "sampled_fisher.score_grad_path",
    "sampled_fisher.exact_fisher_check",
    "empirical_fisher.accumulation",
    "per_example_gradient.accumulation",
    "metric.accumulation",
    "metric.block_schedule",
    "sqrt_metric.lanczos_iterations",
    "metric_inner.multi_rhs",
    "vectorization.mode",
    "vectorization.batch_size",
    "batch.hvp_row_batch_size",
    "batch.fisher_sample_batch_size",
    "batch.empirical_example_batch_size",
    "batch.per_example_block_size",
    "batch.ggn_batch_size",
    "batch.data_microbatch_size",
    "chunk.token_block_size",
    "chunk.class_block_size_with_exact_global_normalization",
    "chunk.output_cotangent_block_size",
    "chunk.parameter_block_size",
    "chunk.layer_block_size",
    "chunk.lm_head_weight_chunk_bytes",
    "schedule.per_example",
    "schedule.per_token",
    "schedule.gradient_accumulation",
    "input.batch_layout",
    "input.length_grouping",
    "teacher_outputs",
    "input.host_to_device",
    "input.residency",
    "memory.vector_residency",
    "memory.intermediate_residency",
    "memory.factor_residency",
    "memory.primal_outputs",
    "memory.jvp_outputs",
    "memory.output_cotangents",
    "memory.output_buffers",
    "fusion.norm",
    "fusion.mlp",
    "fusion.rope",
    "fusion.logits",
    "fusion.loss",
    "layout.contiguity",
    "layout.flatten_order",
    "layout.vector_ops",
    "layout.params",
    "layout.vector",
    "layout.output",
    "layout.aliasing",
    "layout.parametrizations",
    "call.path",
    "call.params",
    "call.buffers",
    "call.tied_weights",
    "call.parametrizations",
    "call.buffer_mutation",
    "call.grad_mode",
    "call.return_type",
    "inverse_metric.iteration_budget",
    "inverse_metric.preconditioner",
    "inverse_metric.factor_reuse",
    "inverse_metric.block_schedule",
    "inverse_metric.multi_rhs",
    "inverse_metric_inner.multi_rhs",
    "composition.child_evaluation",
    "composition.validation",
    "compile.enabled",
    "compile.boundary",
    "compile.backend",
    "compile.mode",
    "compile.fullgraph",
    "compile.dynamic",
    "compile.compiled_autograd",
    "compile.options.epilogue_fusion",
    "compile.options.shape_padding",
    "compile.cuda_graphs",
    "compile.cache_state",
    "activation.recompute",
    "activation.offload",
    "activation.pack_hook",
    "activation.unpack_hook",
    "checkpoint.use_reentrant",
    "checkpoint.early_stop",
    "checkpoint.preserve_rng_state",
    "checkpoint.determinism_check",
    "checkpoint.context_fn",
    "checkpoint.context_fn_callable",
    "checkpoint.moves_to_new_device",
    "checkpoint.uses_global_state",
)
SQRT_METRIC_SPEC_PATHS = {
    "closed_form_factor_square_root": SQRT_METRIC_CLOSED_FORM_PATH,
    "cholesky_factor": SQRT_METRIC_CHOLESKY_PATH,
    "eigenbasis_factor": SQRT_METRIC_EIGENBASIS_PATH,
    "matrix_free_lanczos": SQRT_METRIC_LANCZOS_PATH,
}
SPEC_PATH_TO_RUNTIME = {
    "gradient": {
        "torch_autograd_grad": GRADIENT_PATH,
        "torch_func_grad": GRADIENT_TORCH_FUNC_PATH,
        "torch_func_grad_and_value": GRADIENT_TORCH_FUNC_VALUE_PATH,
        "backward_materialized_grad": GRADIENT_BACKWARD_MATERIALIZED_PATH,
    },
    "jvp": {
        "torch_func_jvp": JVP_PATH,
        "forward_ad_dual": JVP_FORWARD_AD_PATH,
        "torch_func_linearize": JVP_LINEARIZE_PATH,
    },
    "vjp": {
        "torch_func_vjp": VJP_PATH,
        "autograd_grad_outputs": VJP_AUTOGRAD_OUTPUTS_PATH,
        "backward_materialized_grad": VJP_BACKWARD_MATERIALIZED_PATH,
    },
    "hvp": {
        "reverse_over_reverse": HVP_REFERENCE_PATH,
        "autograd_functional_hvp": HVP_FUNCTIONAL_PATH,
        "autograd_functional_vhp": VHP_PATH,
        "jvp_grad": HVP_JVP_GRAD_PATH,
        "forward_ad_dual": HVP_FORWARD_AD_PATH,
        "linearize_grad": HVP_LINEARIZE_GRAD_PATH,
    },
    "ggnvp": {
        "torch_func_jvp": GGN_JVP_HESSIAN_VJP_PATH,
        "forward_ad_dual": GGN_FORWARD_AD_HESSIAN_VJP_PATH,
        "torch_func_linearize": GGN_LINEARIZE_HESSIAN_VJP_PATH,
    },
    "fisher_vp": {
        "streaming_dot_accumulate": FISHER_SCORE_GRADIENT_LOOP_PATH,
        "materialize_score_gradients": FISHER_DENSE_PATH,
        "blockwise_score_matrix": FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    },
    "sampled_fisher_vp": {
        "streaming_dot_accumulate": SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        "materialize_score_gradients": SAMPLED_FISHER_DENSE_PATH,
        "blockwise_score_matrix": SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    },
    "empirical_fisher_vp": {
        "torch_autograd_grad_loop": EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        "torch_func_grad": EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
        "vmap_grad": EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
        "backward_materialized_grad": EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
        "blockwise_gradient_matrix": EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH,
    },
    "per_example_gradient": {
        "torch_autograd_grad_loop": PER_EXAMPLE_GRADIENT_LOOP_PATH,
        "torch_func_grad": PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
        "vmap_grad": PER_EXAMPLE_GRADIENT_VMAP_PATH,
        "backward_materialized_grad": PER_EXAMPLE_GRADIENT_BACKWARD_PATH,
    },
    "metric": {
        "dense_matmul": METRIC_DENSE_PATH,
        "factorized_multiply": METRIC_FACTORIZED_PATH,
        "blockwise_multiply": METRIC_BLOCKWISE_PATH,
        "streaming_multiply": METRIC_STREAMING_PATH,
    },
    "sqrt_metric": SQRT_METRIC_SPEC_PATHS,
    "inverse_sqrt_metric": SQRT_METRIC_SPEC_PATHS,
    "metric_inner": {
        "multiply_then_reduce": METRIC_INNER_MULTIPLY_REDUCE_PATH,
        "factored_gram": METRIC_INNER_FACTORED_GRAM_PATH,
        "sqrt_apply_reduce": METRIC_INNER_SQRT_REDUCE_PATH,
    },
    "inverse_metric": {
        "dense_solve": INVERSE_METRIC_DENSE_PATH,
        "conjugate_gradient": INVERSE_METRIC_CG_PATH,
        "cholesky_solve": INVERSE_METRIC_CHOLESKY_PATH,
        "eigh_solve": INVERSE_METRIC_EIGH_PATH,
        "svd_solve": INVERSE_METRIC_SVD_PATH,
        "factorized_solve": INVERSE_METRIC_FACTORIZED_PATH,
        "blockwise_solve": INVERSE_METRIC_BLOCKWISE_PATH,
        "woodbury_low_rank_solve": INVERSE_METRIC_WOODBURY_PATH,
    },
    "inverse_metric_inner": {
        "solve_then_reduce": INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
        "factored_gram": INVERSE_METRIC_INNER_FACTORED_GRAM_PATH,
        "sqrt_apply_reduce": INVERSE_METRIC_INNER_SQRT_REDUCE_PATH,
    },
    "composition": {
        "materialize_each_child": COMPOSITION_PATH,
        "stream_child_outputs": COMPOSITION_PATH,
        "fuse_adjacent_children": COMPOSITION_PATH,
        "compile_whole_composition": COMPOSITION_PATH,
    },
}
FISHER_STREAMING_PATH_BY_SCORE_GRAD = {
    "torch_autograd_grad_loop": FISHER_SCORE_GRADIENT_LOOP_PATH,
    "torch_func_grad": FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    "vmap_grad": FISHER_SCORE_GRADIENT_VMAP_PATH,
    "backward_materialized_grad": FISHER_BACKWARD_MATERIALIZED_PATH,
}
SAMPLED_FISHER_STREAMING_PATH_BY_SCORE_GRAD = {
    "torch_autograd_grad_loop": SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
    "torch_func_grad": SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    "vmap_grad": SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    "backward_materialized_grad": SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
}
BACKEND_SETTINGS = (
    "autocast",
    "numeric.float32_matmul_precision",
    "numeric.bf16_reduced_precision_reduction",
    "numeric.fp16_reduced_precision_reduction",
    "numeric.deterministic_algorithms",
)
LOSS_SCALING_SETTINGS = (
    "numeric.loss_scaling",
    "numeric.loss_scale",
    "numeric.loss_unscale_degree",
)
SUPPORTED_STANDARD_SETTINGS = (
    *RUNTIME_DTYPE_SETTINGS,
    *SPEC_PATH_KEYS.values(),
    *SPEC_ADDITIONAL_RUNTIME_SETTINGS,
    *BACKEND_SETTINGS,
    *LOSS_SCALING_SETTINGS,
    *FUNCTIONAL_CALL_FIELDS,
    *TORCH_FUNC_FIELDS,
    "vectorization.vmap_chunk_size",
    "vectorization.in_dims",
)
MATRIX_DIMS = 2
FINITE_CHECKS_ENABLED = [True]
BACKEND_SETTINGS_ENABLED = [True]
STANDARD_ANCHOR_PATHS = {
    "gradient": GRADIENT_PATH,
    "jvp": JVP_PATH,
    "vjp": VJP_PATH,
    "hvp": HVP_REFERENCE_PATH,
    "ggnvp": GGN_JVP_HESSIAN_VJP_PATH,
    "fisher_vp": FISHER_SCORE_GRADIENT_LOOP_PATH,
    "sampled_fisher_vp": SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
    "empirical_fisher_vp": EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
    "per_example_gradient": PER_EXAMPLE_GRADIENT_LOOP_PATH,
    "metric": METRIC_DENSE_PATH,
    "sqrt_metric": SQRT_METRIC_EIGENBASIS_PATH,
    "inverse_sqrt_metric": SQRT_METRIC_EIGENBASIS_PATH,
    "metric_inner": METRIC_INNER_MULTIPLY_REDUCE_PATH,
    "inverse_metric": INVERSE_METRIC_DENSE_PATH,
    "inverse_metric_inner": INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
}
VMAP_RUNTIME_PATHS = (
    FISHER_SCORE_GRADIENT_VMAP_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
)
FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = (
    FISHER_SCORE_GRADIENT_LOOP_PATH,
    FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    FISHER_BACKWARD_MATERIALIZED_PATH,
)
FISHER_SCORE_GRADIENT_PRODUCT_PATHS = (
    *FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
    FISHER_SCORE_GRADIENT_VMAP_PATH,
)
SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = (
    SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
)
SAMPLED_FISHER_SCORE_GRADIENT_PRODUCT_PATHS = (
    *SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
    SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
)
EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = (
    EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
    EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
    EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
)
EMPIRICAL_FISHER_GRADIENT_PRODUCT_PATHS = (
    *EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
    EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
)
SCORE_MATRIX_COMPILE_PATH_VALUES = (
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "vmap_grad",
    "backward_materialized_grad",
)
SCORE_MATRIX_COMPILE_ROWS = {
    "fisher_vp": ("fisher_score_grad", "fisher.score_grad_path"),
    "sampled_fisher_vp": (
        "sampled_fisher_score_grad",
        "sampled_fisher.score_grad_path",
    ),
    "empirical_fisher_vp": (
        "empirical_fisher_example_grad",
        "empirical_fisher.grad_path",
    ),
    "per_example_gradient": (
        "per_example_gradient",
        "per_example_gradient.grad_path",
    ),
}
HVP_VECTOR_LOOP_PATHS = (
    HVP_REFERENCE_PATH,
    HVP_FUNCTIONAL_PATH,
    HVP_JVP_GRAD_PATH,
    HVP_FORWARD_AD_PATH,
    HVP_LINEARIZE_GRAD_PATH,
    VHP_PATH,
)
HVP_VECTOR_VMAP_PATHS = (HVP_LINEARIZE_GRAD_PATH,)
INVERSE_METRIC_DIRECT_SOLVE_PATHS = (
    INVERSE_METRIC_DENSE_PATH,
    INVERSE_METRIC_CHOLESKY_PATH,
    INVERSE_METRIC_EIGH_PATH,
    INVERSE_METRIC_SVD_PATH,
)
INVERSE_METRIC_FACTOR_REUSE_PATHS = (
    *INVERSE_METRIC_DIRECT_SOLVE_PATHS,
    INVERSE_METRIC_CG_PATH,
    INVERSE_METRIC_FACTORIZED_PATH,
    INVERSE_METRIC_BLOCKWISE_PATH,
    INVERSE_METRIC_WOODBURY_PATH,
)
JVP_VECTOR_VMAP_PATHS = (JVP_PATH, JVP_LINEARIZE_PATH)
VJP_VECTOR_VMAP_PATHS = (VJP_PATH,)
GGN_VECTOR_VMAP_PATHS = (
    GGN_JVP_HESSIAN_VJP_PATH,
    GGN_LINEARIZE_HESSIAN_VJP_PATH,
)
FISHER_VECTOR_VMAP_PATHS = (
    FISHER_DENSE_PATH,
    *FISHER_SCORE_GRADIENT_PRODUCT_PATHS,
    FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
)
SAMPLED_FISHER_VECTOR_VMAP_PATHS = (
    SAMPLED_FISHER_DENSE_PATH,
    *SAMPLED_FISHER_SCORE_GRADIENT_PRODUCT_PATHS,
    SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
)
EMPIRICAL_FISHER_VECTOR_VMAP_PATHS = (
    EMPIRICAL_FISHER_DENSE_PATH,
    *EMPIRICAL_FISHER_GRADIENT_PRODUCT_PATHS,
    EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH,
)
VECTOR_VMAP_RUNTIME_PATHS = {
    "jvp": JVP_VECTOR_VMAP_PATHS,
    "vjp": VJP_VECTOR_VMAP_PATHS,
    "hvp": HVP_VECTOR_VMAP_PATHS,
    "ggnvp": GGN_VECTOR_VMAP_PATHS,
    "fisher_vp": FISHER_VECTOR_VMAP_PATHS,
    "sampled_fisher_vp": SAMPLED_FISHER_VECTOR_VMAP_PATHS,
    "empirical_fisher_vp": EMPIRICAL_FISHER_VECTOR_VMAP_PATHS,
    "metric_inner": (
        METRIC_INNER_MULTIPLY_REDUCE_PATH,
        METRIC_INNER_FACTORED_GRAM_PATH,
        METRIC_INNER_SQRT_REDUCE_PATH,
    ),
    "inverse_metric_inner": (
        INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
        INVERSE_METRIC_INNER_FACTORED_GRAM_PATH,
        INVERSE_METRIC_INNER_SQRT_REDUCE_PATH,
    ),
    "composition": (COMPOSITION_PATH,),
}
VECTOR_LOOP_RUNTIME_PATHS = {
    "jvp": (JVP_PATH, JVP_FORWARD_AD_PATH, JVP_LINEARIZE_PATH),
    "vjp": (
        VJP_PATH,
        VJP_AUTOGRAD_OUTPUTS_PATH,
        VJP_BACKWARD_MATERIALIZED_PATH,
    ),
    "hvp": HVP_VECTOR_LOOP_PATHS,
    "ggnvp": (
        GGN_DENSE_PATH,
        GGN_JVP_HESSIAN_VJP_PATH,
        GGN_FORWARD_AD_HESSIAN_VJP_PATH,
        GGN_LINEARIZE_HESSIAN_VJP_PATH,
    ),
    "fisher_vp": (
        FISHER_DENSE_PATH,
        FISHER_SCORE_GRADIENT_LOOP_PATH,
        FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        FISHER_SCORE_GRADIENT_VMAP_PATH,
        FISHER_BACKWARD_MATERIALIZED_PATH,
        FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    ),
    "sampled_fisher_vp": (
        SAMPLED_FISHER_DENSE_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
        SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
        SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    ),
    "empirical_fisher_vp": (
        EMPIRICAL_FISHER_DENSE_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
        EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
        EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH,
    ),
    "inverse_metric": (
        INVERSE_METRIC_DENSE_PATH,
        INVERSE_METRIC_CG_PATH,
        INVERSE_METRIC_CHOLESKY_PATH,
        INVERSE_METRIC_EIGH_PATH,
        INVERSE_METRIC_SVD_PATH,
        INVERSE_METRIC_FACTORIZED_PATH,
        INVERSE_METRIC_BLOCKWISE_PATH,
        INVERSE_METRIC_WOODBURY_PATH,
    ),
    "metric_inner": (
        METRIC_INNER_MULTIPLY_REDUCE_PATH,
        METRIC_INNER_FACTORED_GRAM_PATH,
        METRIC_INNER_SQRT_REDUCE_PATH,
    ),
    "inverse_metric_inner": (
        INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
        INVERSE_METRIC_INNER_FACTORED_GRAM_PATH,
        INVERSE_METRIC_INNER_SQRT_REDUCE_PATH,
    ),
    "composition": (COMPOSITION_PATH,),
}
SPEC_REQUIRED_PATH_OPERATORS = {
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
}
ACTIVE_CHECKPOINT_SETTINGS = (
    "checkpoint_non_reentrant_by_layer",
    "checkpoint_selective",
)
ActivationPackHooks = Mapping[str, Callable[[torch.Tensor], Any]]
ActivationUnpackHooks = Mapping[str, Callable[[Any], torch.Tensor]]
CheckpointContextFns = Mapping[str, Callable[[], Any]]
MMapResidency = Callable[[torch.Tensor, str], torch.Tensor]
IntermediateTransform = Callable[[TensorTree], TensorTree]


def _is_tensor_tree_dict(value: TensorTree) -> TypeGuard[dict[str, TensorTree]]:
    return isinstance(value, dict)


def _is_tensor_tree_tuple(value: TensorTree) -> TypeGuard[tuple[TensorTree, ...]]:
    return isinstance(value, tuple)


def checkpoint_operation(
    candidate: Candidate,
    function: Callable[..., TensorTree],
    args: Sequence[Any],
    *,
    policy_key: str,
    activation_pack_hooks: ActivationPackHooks | None = None,
    activation_unpack_hooks: ActivationUnpackHooks | None = None,
    checkpoint_contexts: CheckpointContextFns | None = None,
) -> CandidateOperation:
    """Return direct or checkpointed execution for an adapter operation.

    Raises:
        AdmissionError: If the candidate has missing or rejected checkpoint fields.
    """
    setting = _checkpoint_setting(candidate, policy_key)
    offload = _activation_offload(candidate)

    if setting == "none":
        return _with_activation_offload(
            candidate,
            _direct_operation(function, args),
            offload,
            activation_pack_hooks,
            activation_unpack_hooks,
        )

    if setting not in ACTIVE_CHECKPOINT_SETTINGS:
        message = f"checkpoint setting is unsupported: {setting}"
        raise AdmissionError(message)

    admit_checkpoint(candidate.settings)
    context_fn = _checkpoint_context_fn(candidate, checkpoint_contexts)

    def operation() -> TensorTree:
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=_checkpoint_bool(
                candidate,
                "checkpoint.preserve_rng_state",
            ),
            determinism_check=candidate.settings["checkpoint.determinism_check"],
            context_fn=context_fn,
            early_stop=_checkpoint_bool(candidate, "checkpoint.early_stop"),
        )

    return _with_activation_offload(
        candidate,
        operation,
        offload,
        activation_pack_hooks,
        activation_unpack_hooks,
    )


def _checkpoint_setting(candidate: Candidate, policy_key: str) -> str:
    if policy_key not in candidate.settings:
        message = f"checkpoint policy key is missing: {policy_key}"
        raise AdmissionError(message)

    setting = candidate.settings[policy_key]

    if not isinstance(setting, str):
        message = f"checkpoint setting must be a string: {policy_key}"
        raise AdmissionError(message)

    return setting


def _direct_operation(
    function: Callable[..., TensorTree],
    args: Sequence[Any],
) -> CandidateOperation:
    def operation() -> TensorTree:
        return function(*args)

    return operation


def _with_activation_offload(
    candidate: Candidate,
    operation: CandidateOperation,
    offload: str,
    activation_pack_hooks: ActivationPackHooks | None,
    activation_unpack_hooks: ActivationUnpackHooks | None,
) -> CandidateOperation:
    if offload == "none":
        return operation

    pack_hook, unpack_hook = _saved_tensor_hooks(
        candidate,
        offload,
        activation_pack_hooks,
        activation_unpack_hooks,
    )

    def wrapped() -> TensorTree:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            return operation()

    return wrapped


def _activation_offload(candidate: Candidate) -> str:
    value = candidate.settings.get("activation.offload")

    if value not in {"none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}:
        message = "activation.offload is invalid"
        raise AdmissionError(message)

    return value


def _saved_tensor_hooks(
    candidate: Candidate,
    offload: str,
    activation_pack_hooks: ActivationPackHooks | None,
    activation_unpack_hooks: ActivationUnpackHooks | None,
) -> tuple[Callable[[torch.Tensor], Any], Callable[[Any], torch.Tensor]]:
    if offload == "saved_tensor_hooks_cpu":
        return _cpu_pack_hook, _cpu_unpack_hook

    pack_hook_id = candidate.settings.get("activation.pack_hook")
    unpack_hook_id = candidate.settings.get("activation.unpack_hook")

    if not isinstance(pack_hook_id, str) or not isinstance(unpack_hook_id, str):
        message = "custom_saved_tensor_hooks requires activation pack and unpack hooks"
        raise AdmissionError(message)

    if activation_pack_hooks is None or pack_hook_id not in activation_pack_hooks:
        message = f"activation pack hook is not registered: {pack_hook_id}"
        raise AdmissionError(message)

    if activation_unpack_hooks is None or unpack_hook_id not in activation_unpack_hooks:
        message = f"activation unpack hook is not registered: {unpack_hook_id}"
        raise AdmissionError(message)

    return activation_pack_hooks[pack_hook_id], activation_unpack_hooks[unpack_hook_id]


def _cpu_pack_hook(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
    return tensor.detach().cpu(), tensor.device


def _cpu_unpack_hook(packed: tuple[torch.Tensor, torch.device]) -> torch.Tensor:
    tensor, device = packed

    return tensor.to(device)


def _checkpoint_context_fn(
    candidate: Candidate,
    checkpoint_contexts: CheckpointContextFns | None,
) -> Callable[[], Any]:
    context_fn = candidate.settings["checkpoint.context_fn"]

    if context_fn == "none":
        return noop_context_fn

    context_id = candidate.settings.get("checkpoint.context_fn_callable")

    if not isinstance(context_id, str):
        message = "checkpoint.context_fn=declared_context_pair requires context id"
        raise AdmissionError(message)

    if checkpoint_contexts is None or context_id not in checkpoint_contexts:
        message = f"checkpoint context is not registered: {context_id}"
        raise AdmissionError(message)

    return checkpoint_contexts[context_id]


def _checkpoint_bool(candidate: Candidate, key: str) -> bool:
    value = candidate.settings[key]

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be false or true"
    raise AdmissionError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionChild:
    """Child operator reference used by sequential composition."""

    name: str
    candidate: Candidate
    component: Callable[[Batch, TensorTree], TensorTree]
    anchor_component: Callable[[Batch, TensorTree], TensorTree]
    reference_check: ReferenceCheck
    input_signature: Mapping[str, Any]


def composition_operation_factory(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    children: Sequence[CompositionChild] = (),
) -> OperationFactory:
    """Return an operation factory for sequential operator composition."""
    expression = _operator_composition_expression(operator)
    sequential_order = _composition_expression_sequential_order(expression)
    order_tuple = (
        sequential_order
        if sequential_order is not None
        else _operator_composition_order(operator)
    )
    component_map = dict(components)
    fused_component_map = {} if fused_components is None else dict(fused_components)
    child_map = _composition_child_map(order_tuple, children)
    _require_composition_components(order_tuple, component_map)
    expression = None if sequential_order is not None else expression

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        _require_candidate_family(operator, candidate)
        _require_supported_standard_settings(operator, candidate)
        _require_path(
            operator.kind,
            _runtime_path(operator, candidate),
            (COMPOSITION_PATH,),
        )
        executable_components = _composition_execution_components(
            candidate.settings,
            order_tuple,
            component_map,
            child_map,
        )
        fused_component = _fused_composition_component(
            candidate.settings,
            order_tuple,
            fused_component_map,
        )
        _require_loss_scaling_settings(operator, candidate.settings)
        _require_composition_execution_settings(candidate.settings)
        executable_components = _compile_composition_child_components(
            candidate.settings,
            order_tuple,
            executable_components,
            batch,
            vector,
        )
        output_buffer = _composition_output_buffer(candidate.settings, vector)

        def operation() -> TensorTree:
            runtime_batch = _runtime_batch(batch, candidate.settings)
            result = _runtime_vector(vector, candidate.settings)

            def run_components() -> TensorTree:
                if expression is not None:
                    return _run_composition_expression(
                        candidate.settings,
                        expression,
                        executable_components,
                        runtime_batch,
                        result,
                    )

                return _run_composition_components(
                    candidate.settings,
                    order_tuple,
                    executable_components,
                    fused_component,
                    runtime_batch,
                    result,
                )

            def run_scaled_components() -> TensorTree:
                component_output = run_components()
                scaled_output = _loss_scaled_output_source(
                    operator,
                    candidate.settings,
                    component_output,
                )

                return _loss_unscaled_output(
                    operator,
                    candidate.settings,
                    scaled_output,
                )

            return _run_with_backend_settings(
                candidate.settings,
                lambda: _run_with_call_grad_mode(
                    candidate.settings,
                    lambda: _runtime_output_to_buffer(
                        _runtime_output(
                            run_scaled_components(),
                            candidate.settings,
                        ),
                        output_buffer,
                    ),
                ),
            )

        return _compile_operation(operator, candidate.settings, operation)

    return factory


def composition_reference_check(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    thresholds: Mapping[str, float],
    numeric_bound_fields: Mapping[str, Any] | None = None,
    children: Sequence[CompositionChild] = (),
) -> ReferenceCheck:
    """Return a reference check for sequential operator composition.

    Raises:
        MaterializationError: If thresholds are empty.
    """
    if not thresholds:
        message = "composition reference thresholds are required"
        raise MaterializationError(message)

    expression = _operator_composition_expression(operator)
    sequential_order = _composition_expression_sequential_order(expression)
    order_tuple = (
        sequential_order
        if sequential_order is not None
        else _operator_composition_order(operator)
    )
    component_map = dict(components)
    anchor_component_map = dict(anchor_components)
    fused_component_map = {} if fused_components is None else dict(fused_components)
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    child_map = _composition_child_map(order_tuple, children)
    _require_composition_components(order_tuple, component_map)
    _require_composition_components(order_tuple, anchor_component_map)
    expression = None if sequential_order is not None else expression

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        try:
            anchor_candidate = dataclasses.replace(
                candidate,
                settings=_anchor_settings(operator, candidate, COMPOSITION_PATH),
            )
            candidate_components, anchor_components_for_row = (
                _composition_reference_components(
                    candidate.settings,
                    order_tuple,
                    component_map,
                    anchor_component_map,
                    child_map,
                )
            )
            reference_children = _composition_reference_children(
                candidate.settings,
                order_tuple,
                child_map,
            )
            candidate_output, anchor_output, component_errors, child_results = (
                _composition_reference_outputs(
                    operator,
                    candidate,
                    anchor_candidate,
                    batch,
                    vector,
                    order_tuple,
                    candidate_components,
                    anchor_components_for_row,
                    fused_component_map,
                    reference_children,
                    expression,
                )
            )
        except (MaterializationError, ReferenceFailedError) as error:
            raise ReferenceFailedError(str(error)) from error

        measurements = tree_error_measurements(candidate_output, anchor_output)
        _merge_component_measurements(measurements, component_errors)
        measurements.update(
            _semantic_measurements(operator, batch, vector, candidate_output)
        )
        effective_thresholds = _reference_thresholds_for_operator(operator, thresholds)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)
        _apply_numeric_error_bound(
            measurements,
            effective_thresholds,
            candidate.settings,
            bound_fields,
            anchor_output,
        )

        return ReferenceResult(
            "composition_anchor",
            effective_thresholds,
            measurements,
            child_results=child_results,
        )

    return check


def _composition_reference_outputs(
    operator: OperatorSpec,
    candidate: Candidate,
    anchor_candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ],
    children: Mapping[str, CompositionChild],
    expression: Mapping[str, Any] | None,
) -> tuple[
    TensorTree,
    TensorTree,
    dict[str, dict[str, float]],
    tuple[ReferenceChildResult, ...],
]:
    _require_candidate_family(operator, candidate)
    _require_candidate_family(operator, anchor_candidate)
    _require_supported_standard_settings(operator, candidate)
    _require_supported_standard_settings(operator, anchor_candidate)
    _require_composition_execution_settings(candidate.settings)
    _require_loss_scaling_settings(operator, candidate.settings)
    _require_path(
        operator.kind,
        _runtime_path(operator, candidate),
        (COMPOSITION_PATH,),
    )
    _require_path(
        operator.kind,
        _runtime_path(operator, anchor_candidate),
        (COMPOSITION_PATH,),
    )
    candidate_batch = _runtime_batch(batch, candidate.settings)
    anchor_batch = _runtime_batch(batch, anchor_candidate.settings)
    candidate_result = _runtime_vector(vector, candidate.settings)
    anchor_result = _runtime_vector(vector, anchor_candidate.settings)
    fused_component = _fused_composition_component(
        candidate.settings,
        order,
        fused_components,
    )
    fused_result = None
    component_errors = {}
    child_results = []

    if expression is not None:
        (
            candidate_result,
            anchor_result,
            component_errors,
            child_results,
        ) = _composition_expression_reference_outputs(
            candidate.settings,
            expression,
            candidate_batch,
            anchor_batch,
            candidate_result,
            anchor_result,
            components,
            anchor_components,
            children,
        )
        candidate_result = _loss_scaled_output_source(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = _loss_unscaled_output(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = _runtime_output(candidate_result, candidate.settings)
        anchor_result = _runtime_output(anchor_result, anchor_candidate.settings)

        return candidate_result, anchor_result, component_errors, tuple(child_results)

    if fused_component is not None:
        fused_result = fused_component(candidate_batch, candidate_result)

    for component_name in order:
        child = children.get(component_name)

        if child is not None:
            child_input_signature = {
                **dict(child.input_signature),
                "component": component_name,
                "component_input": tree_signature(candidate_result),
            }
            child_result = child.reference_check(
                child.candidate,
                candidate_batch,
                candidate_result,
            )
            child_results.append(
                ReferenceChildResult(
                    component_name,
                    child.candidate,
                    child_input_signature,
                    child_result,
                )
            )

        candidate_result = components[component_name](
            candidate_batch,
            candidate_result,
        )
        anchor_result = anchor_components[component_name](
            anchor_batch,
            anchor_result,
        )
        component_errors[component_name] = tree_error_measurements(
            candidate_result,
            anchor_result,
        )

    if fused_result is not None:
        component_errors["fused_composition"] = tree_error_measurements(
            fused_result,
            candidate_result,
        )
        candidate_result = fused_result

    candidate_result = _loss_scaled_output_source(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = _loss_unscaled_output(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = _runtime_output(candidate_result, candidate.settings)
    anchor_result = _runtime_output(anchor_result, anchor_candidate.settings)

    return candidate_result, anchor_result, component_errors, tuple(child_results)


def _composition_execution_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, CompositionChild],
) -> Mapping[str, Callable[[Batch, TensorTree], TensorTree]]:
    if settings.get("composition.execution") == "fuse_adjacent_children":
        return components

    mode = _composition_child_evaluation(settings)

    if mode == "inline_child_lowering":
        return components

    child_components = {name: child.component for name, child in children.items()}
    _require_composition_components(order, child_components)

    return child_components


def _fused_composition_component(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ],
) -> Callable[[Batch, TensorTree], TensorTree] | None:
    if settings.get("composition.execution") != "fuse_adjacent_children":
        return None

    component = fused_components.get(order)

    if component is None:
        message = "fuse_adjacent_children requires a fused component for child order"
        raise MaterializationError(message)

    return component


def _operator_composition_expression(
    operator: OperatorSpec,
) -> Mapping[str, Any] | None:
    expression = operator.semantics.get("combine")

    if expression is None:
        return None

    if not isinstance(expression, Mapping):
        message = "composition combine expression must be a mapping"
        raise MaterializationError(message)

    _require_composition_expression(expression)

    return expression


def _composition_expression_sequential_order(
    expression: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    if expression is None:
        return None

    kind = _composition_expression_kind(expression)

    if kind == "child":
        return (_composition_expression_child(expression),)

    if kind != "compose":
        return None

    terms = _composition_expression_terms(expression)

    if not all(_composition_expression_kind(term) == "child" for term in terms):
        return None

    return tuple(_composition_expression_child(term) for term in reversed(terms))


def _require_composition_expression(expression: Mapping[str, Any]) -> None:
    kind = _composition_expression_kind(expression)

    if kind in {"child", "source"}:
        _composition_expression_child(expression)

        return

    if kind == "scaled_identity":
        _composition_expression_coefficient(expression)

        return

    if kind == "compose":
        for term in _composition_expression_terms(expression):
            _require_composition_expression(term)

        return

    if kind == "linear_combination":
        for term in _composition_expression_weighted_terms(expression):
            _require_composition_expression(term["term"])

        return

    message = f"composition expression kind is unsupported: {kind}"
    raise MaterializationError(message)


def _composition_expression_kind(expression: Mapping[str, Any]) -> str:
    kind = expression.get("kind")

    if isinstance(kind, str) and kind:
        return kind

    message = "composition expression kind must be a non-empty string"
    raise MaterializationError(message)


def _composition_expression_child(expression: Mapping[str, Any]) -> str:
    child = expression.get("name")

    if child is None:
        child = expression.get("child")

    if isinstance(child, str) and child:
        return child

    message = "composition expression child must be a non-empty string"
    raise MaterializationError(message)


def _composition_expression_coefficient(expression: Mapping[str, Any]) -> float:
    coefficient = expression.get("coefficient")

    if isinstance(coefficient, int | float) and not isinstance(coefficient, bool):
        return float(coefficient)

    message = "composition expression coefficient must be a number"
    raise MaterializationError(message)


def _composition_expression_terms(
    expression: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    terms = expression.get("terms")

    if (
        isinstance(terms, Sequence)
        and not isinstance(terms, str)
        and terms
        and all(isinstance(term, Mapping) for term in terms)
    ):
        return tuple(terms)

    message = "composition expression terms must be non-empty mappings"
    raise MaterializationError(message)


def _composition_expression_weighted_terms(
    expression: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    terms = expression.get("terms")

    if not isinstance(terms, Sequence) or isinstance(terms, str) or not terms:
        message = "linear composition terms must be non-empty mappings"
        raise MaterializationError(message)

    normalized = []

    for term in terms:
        if not isinstance(term, Mapping):
            message = "linear composition term must be a mapping"
            raise MaterializationError(message)

        coefficient = term.get("coefficient")
        nested = term.get("term")

        if not isinstance(coefficient, int | float) or isinstance(coefficient, bool):
            message = "linear composition coefficient must be a number"
            raise MaterializationError(message)

        if not isinstance(nested, Mapping):
            message = "linear composition term expression must be a mapping"
            raise MaterializationError(message)

        normalized.append({"coefficient": float(coefficient), "term": nested})

    return tuple(normalized)


def _run_composition_expression(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    if settings.get("composition.execution") == "fuse_adjacent_children":
        message = "fuse_adjacent_children requires a sequential compose expression"
        raise MaterializationError(message)

    result, _, _ = _evaluate_composition_expression(
        settings,
        expression,
        components,
        batch,
        vector,
        children={},
        run_child_checks=False,
    )

    return result


def _composition_expression_reference_outputs(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    candidate_batch: Batch,
    anchor_batch: Batch,
    candidate_vector: TensorTree,
    anchor_vector: TensorTree,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, CompositionChild],
) -> tuple[
    TensorTree,
    TensorTree,
    dict[str, dict[str, float]],
    tuple[ReferenceChildResult, ...],
]:
    run_child_checks = _composition_validation(settings) == "validate_each_child"
    candidate_result, candidate_outputs, child_results = (
        _evaluate_composition_expression(
            settings,
            expression,
            components,
            candidate_batch,
            candidate_vector,
            children=children,
            run_child_checks=run_child_checks,
        )
    )
    anchor_result, anchor_outputs, _ = _evaluate_composition_expression(
        settings,
        expression,
        anchor_components,
        anchor_batch,
        anchor_vector,
        children={},
        run_child_checks=False,
    )
    component_errors = _composition_expression_component_errors(
        candidate_outputs,
        anchor_outputs,
    )

    return candidate_result, anchor_result, component_errors, child_results


def _evaluate_composition_expression(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
    *,
    children: Mapping[str, CompositionChild],
    run_child_checks: bool,
) -> tuple[
    TensorTree,
    tuple[tuple[str, TensorTree], ...],
    tuple[ReferenceChildResult, ...],
]:
    leaf_outputs = []
    child_results = []

    def evaluate(term: Mapping[str, Any], current_vector: TensorTree) -> TensorTree:
        kind = _composition_expression_kind(term)

        if kind == "child":
            child = _composition_expression_child(term)
            result = components[child](batch, current_vector)
            leaf_outputs.append((child, result))

            if run_child_checks and child in children:
                child_results.append(
                    _composition_expression_child_result(
                        child,
                        children[child],
                        batch,
                        current_vector,
                    )
                )

            return result

        if kind == "source":
            child = _composition_expression_child(term)
            result = components[child](batch, {})
            leaf_outputs.append((f"source:{child}", result))

            if run_child_checks and child in children:
                child_results.append(
                    _composition_expression_child_result(
                        child,
                        children[child],
                        batch,
                        {},
                    )
                )

            return result

        if kind == "scaled_identity":
            return _tree_scale_runtime(
                settings,
                current_vector,
                _composition_expression_coefficient(term),
            )

        if kind == "compose":
            terms = _composition_expression_terms(term)
            result = current_vector
            execution_terms = tuple(reversed(terms))

            for index, nested in enumerate(execution_terms):
                result = evaluate(nested, result)
                result = _composition_intermediate_residency(
                    settings,
                    result,
                    is_last=index == len(execution_terms) - 1,
                )

            return result

        if kind == "linear_combination":
            result = None

            for weighted in _composition_expression_weighted_terms(term):
                term_output = evaluate(weighted["term"], current_vector)
                scaled = _tree_scale_runtime(
                    settings,
                    term_output,
                    weighted["coefficient"],
                )

                if result is None:
                    result = scaled
                else:
                    result = _tree_add_runtime(settings, result, scaled)

            if result is None:
                message = "linear composition produced no terms"
                raise MaterializationError(message)

            return result

        message = f"composition expression kind is unsupported: {kind}"
        raise MaterializationError(message)

    result = evaluate(expression, vector)

    return result, tuple(leaf_outputs), tuple(child_results)


def _composition_expression_child_result(
    name: str,
    child: CompositionChild,
    batch: Batch,
    vector: TensorTree,
) -> ReferenceChildResult:
    child_input_signature = {
        **dict(child.input_signature),
        "component": name,
        "component_input": tree_signature(vector),
    }
    child_result = child.reference_check(child.candidate, batch, vector)

    return ReferenceChildResult(
        name,
        child.candidate,
        child_input_signature,
        child_result,
    )


def _composition_expression_component_errors(
    candidate_outputs: tuple[tuple[str, TensorTree], ...],
    anchor_outputs: tuple[tuple[str, TensorTree], ...],
) -> dict[str, dict[str, float]]:
    if len(candidate_outputs) != len(anchor_outputs):
        message = "composition candidate and anchor component counts differ"
        raise MaterializationError(message)

    errors = {}

    for index, (candidate, anchor) in enumerate(
        zip(candidate_outputs, anchor_outputs, strict=True)
    ):
        candidate_name, candidate_output = candidate
        anchor_name, anchor_output = anchor

        if candidate_name != anchor_name:
            message = "composition candidate and anchor component names differ"
            raise MaterializationError(message)

        errors[f"{index}:{candidate_name}"] = tree_error_measurements(
            candidate_output,
            anchor_output,
        )

    return errors


def _run_composition_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_component: Callable[[Batch, TensorTree], TensorTree] | None,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    def vector_runner(selected_vector: TensorTree) -> TensorTree:
        return _run_composition_single_vector(
            settings,
            order,
            components,
            fused_component,
            batch,
            selected_vector,
        )

    return _run_tensor_tree_by_vectorization_mode(vector, settings, vector_runner)


def _run_composition_single_vector(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_component: Callable[[Batch, TensorTree], TensorTree] | None,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    if fused_component is not None:
        return fused_component(batch, vector)

    result = vector

    for index, component_name in enumerate(order):
        result = components[component_name](batch, result)
        result = _composition_intermediate_residency(
            settings,
            result,
            is_last=index == len(order) - 1,
        )

    return result


def _composition_intermediate_residency(
    settings: Mapping[str, Any],
    result: TensorTree,
    *,
    is_last: bool,
) -> TensorTree:
    if is_last:
        return result

    residency = settings.get("memory.intermediate_residency")

    if residency is None:
        return result

    return _tree_residency(result, residency, "memory.intermediate_residency")


def _compile_composition_child_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
) -> Mapping[str, Callable[[Batch, TensorTree], TensorTree]]:
    if settings.get("compile.boundary") != "composition_child":
        return components

    if settings.get("composition.execution") == "fuse_adjacent_children":
        message = "compile.boundary=composition_child requires child calls"
        raise MaterializationError(message)

    _validate_compile_cache_state(settings)
    compiled_components = {}
    warm_result = _composition_child_warm_vector(settings, vector)
    warm_batch = _composition_child_warm_batch(settings, batch)

    for name in order:
        compiled_component = _compiled_composition_component(
            settings,
            components[name],
        )
        compiled_components[name] = compiled_component

        if warm_batch is not None and warm_result is not None:
            warm_result = compiled_component(warm_batch, warm_result)

    return compiled_components


def _composition_child_warm_batch(
    settings: Mapping[str, Any],
    batch: Batch,
) -> Batch | None:
    if settings.get("compile.cache_state") != "warm_cache":
        return None

    return _runtime_batch(batch, settings)


def _composition_child_warm_vector(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("compile.cache_state") != "warm_cache":
        return None

    warm_vector = _runtime_vector(vector, settings)
    mode = settings.get("vectorization.mode")

    if mode == "single_loop":
        vector_in_dims = _vector_tree_in_dims(warm_vector, settings)

        return _vector_tree_select(warm_vector, vector_in_dims, 0)

    if mode == "manual_batch":
        vector_in_dims = _vector_tree_in_dims(warm_vector, settings)
        vector_count = _vector_tree_batch_size(warm_vector, vector_in_dims)
        batch_size = _manual_vector_batch_size(settings)

        return _vector_tree_slice(
            warm_vector,
            vector_in_dims,
            0,
            min(batch_size, vector_count),
        )

    return warm_vector


def _compiled_composition_component(
    settings: Mapping[str, Any],
    component: Callable[[Batch, TensorTree], TensorTree],
) -> Callable[[Batch, TensorTree], TensorTree]:
    return _compiled_callable(settings, component, use_backend_settings=False)


def _composition_reference_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, CompositionChild],
) -> tuple[
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
]:
    mode = _composition_child_evaluation(settings)

    if mode == "inline_child_lowering":
        return components, anchor_components

    child_components = {name: child.component for name, child in children.items()}
    child_anchor_components = {
        name: child.anchor_component for name, child in children.items()
    }
    _require_composition_components(order, child_components)
    _require_composition_components(order, child_anchor_components)

    return child_components, child_anchor_components


def _composition_reference_children(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    children: Mapping[str, CompositionChild],
) -> Mapping[str, CompositionChild]:
    validation = _composition_validation(settings)

    if validation == "validate_composed_output":
        return {}

    _require_composition_children(order, children)

    return children


def _require_composition_execution_settings(settings: Mapping[str, Any]) -> None:
    execution = settings.get("composition.execution")

    if (
        execution == "materialize_each_child"
        and settings.get("composition.child_evaluation") != "selected_child_rows"
    ):
        message = (
            "materialize_each_child requires "
            "composition.child_evaluation=selected_child_rows"
        )
        raise MaterializationError(message)

    if execution != "compile_whole_composition":
        return

    if settings.get("compile.enabled") != "true":
        message = "compile_whole_composition requires compile.enabled=true"
        raise MaterializationError(message)


def _composition_child_evaluation(settings: Mapping[str, Any]) -> str:
    value = settings.get("composition.child_evaluation")

    if value not in {"selected_child_rows", "inline_child_lowering"}:
        message = "composition.child_evaluation is invalid"
        raise MaterializationError(message)

    return value


def _composition_validation(settings: Mapping[str, Any]) -> str:
    value = settings.get("composition.validation")

    if value not in {"validate_each_child", "validate_composed_output"}:
        message = "composition.validation is invalid"
        raise MaterializationError(message)

    return value


def _composition_child_map(
    order: tuple[str, ...],
    children: Sequence[CompositionChild],
) -> dict[str, CompositionChild]:
    child_map = {child.name: child for child in children}

    if len(child_map) != len(children):
        message = "composition child names must be unique"
        raise MaterializationError(message)

    unknown = tuple(name for name in child_map if name not in order)

    if unknown:
        message = f"composition child names are not in the order: {unknown}"
        raise MaterializationError(message)

    return child_map


def _require_composition_children(
    order: tuple[str, ...],
    children: Mapping[str, CompositionChild],
) -> None:
    if set(children) != set(order):
        message = "composition children must match composition order"
        raise MaterializationError(message)


def _merge_component_measurements(
    measurements: dict[str, Any],
    component_errors: Mapping[str, Mapping[str, float]],
) -> None:
    if not component_errors:
        return

    max_abs = float(measurements["max_abs_diff"])
    max_rel = float(measurements["max_rel_diff"])

    for errors in component_errors.values():
        max_abs = max(max_abs, float(errors["max_abs_diff"]))
        max_rel = max(max_rel, float(errors["max_rel_diff"]))

    measurements["max_abs_diff"] = max_abs
    measurements["max_rel_diff"] = max_rel
    measurements["component_errors"] = {
        name: dict(errors) for name, errors in component_errors.items()
    }


def _require_thresholds_for_measurements(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    exact_required = ("psd_violation", "damping_min", "condition_number_max")
    missing = tuple(
        key
        for key in measurements
        if key.endswith(("_diff", "_residual")) or key in exact_required
        if key not in thresholds
    )

    if missing:
        message = f"reference thresholds are missing measurements: {missing}"
        raise ReferenceFailedError(message)


def _require_vhp_reference_policy(
    candidate: Candidate,
    batch: Batch,
    thresholds: Mapping[str, float],
) -> None:
    if candidate.settings.get("hvp.path") != "autograd_functional_vhp":
        return

    required_thresholds = (
        "symmetry_max_abs_diff",
        "directional_abs_diff",
        "directional_rel_diff",
    )
    missing_thresholds = tuple(
        threshold for threshold in required_thresholds if threshold not in thresholds
    )

    if missing_thresholds:
        message = f"vhp reference thresholds are missing: {missing_thresholds}"
        raise ReferenceFailedError(message)

    _batch_tree(batch, "symmetry_vector")


def _semantic_measurements(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
    if operator.kind == "ggnvp":
        loss_hessian = _batch_tensor(batch, "loss_hessian")

        return {
            "symmetry_max_abs_diff": _matrix_symmetry_error(loss_hessian),
            "psd_violation": _matrix_psd_violation(loss_hessian),
        }

    if operator.kind == "metric":
        if _metric_representation_kind(operator) == "matrix_free":
            return {}

        matrix = _metric_dense_matrix(operator, batch, vector)

        return {
            "symmetry_max_abs_diff": _matrix_symmetry_error(matrix),
            "psd_violation": _matrix_psd_violation(matrix),
        }

    if operator.kind == "inverse_metric":
        if _metric_representation_kind(operator) == "matrix_free":
            return {}

        matrix = _metric_dense_matrix(operator, batch, vector)
        inverse_matrix = _inverse_metric_matrix(operator, matrix, batch)
        vector_tensor = _flatten_vector(vector)
        output_tensor = _flatten_vector(output)
        measurements = {
            "symmetry_max_abs_diff": _matrix_symmetry_error(inverse_matrix),
            "psd_violation": _matrix_psd_violation(inverse_matrix),
            "inverse_residual": _inverse_residual(
                inverse_matrix,
                output_tensor,
                vector_tensor,
            ),
            "condition_number_max": _matrix_condition_number(inverse_matrix),
        }
        damping = _inverse_metric_damping(operator)

        if damping > 0.0:
            measurements["damping_min"] = damping

        return measurements

    return {}


def _matrix_free_inverse_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
    if operator.kind != "inverse_metric":
        return {}

    if _metric_representation_kind(operator) != "matrix_free":
        return {}

    damping = _inverse_metric_damping(operator)
    flat_output = _flatten_vector(output)
    applied = _metric_apply_flat(
        operator,
        batch,
        output,
        flat_output,
        damping,
        METRIC_STREAMING_PATH,
        candidate.settings,
    )
    residual = (applied - _flatten_vector(vector)).norm()
    denominator = _flatten_vector(vector).norm()

    if math.isclose(float(denominator.item()), 0.0, rel_tol=0.0, abs_tol=0.0):
        inverse_residual = float(residual.item())
    else:
        inverse_residual = float((residual / denominator).item())

    return {
        "inverse_residual": inverse_residual,
        "damping_min": damping,
    }


def _first_order_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    function_objectives: Mapping[str, FunctionObjective],
) -> dict[str, float]:
    if operator.kind == "gradient":
        scalar = _scalar_objective(operator, scalar_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def scalar_function(active_params: ParameterTree) -> torch.Tensor:
            return scalar(active_params, buffers, batch, context)

        finite_difference = finite_difference_jvp(scalar_function, params, vector)
        directional = _layout_aware_tree_dot(
            candidate.settings,
            candidate_output,
            vector,
        )
        errors = tree_error_measurements(directional, finite_difference)

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "jvp":
        function = _function_objective(operator, function_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            output = function(active_params, buffers, batch, context)

            return _checked_function_output(
                candidate.settings,
                output,
                "function objective output",
            )

        finite_difference = finite_difference_jvp(tensor_function, params, vector)
        errors = _layout_aware_tree_error_measurements(
            candidate,
            candidate_output,
            finite_difference,
        )

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "vjp":
        function = _function_objective(operator, function_objectives)
        tangent = _batch_tree(batch, "tangent_vector")
        _require_min_probe_norm(tangent, "tangent_vector")
        _require_min_probe_norm(vector, "cotangent_vector")
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            output = function(active_params, buffers, batch, context)

            return _checked_function_output(
                candidate.settings,
                output,
                "function objective output",
            )

        return {
            "inner_abs_diff": float(
                vjp_dot_identity_error(tensor_function, params, tangent, vector).item()
            )
        }

    return {}


def _hvp_finite_difference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    anchor_candidate: Candidate,
    candidate_factory: OperationFactory,
    parameter_surface: ParameterSurface | None,
) -> dict[str, float]:
    if operator.kind != "hvp":
        return {}

    scalar = _scalar_objective(operator, scalar_objectives)
    context = ObjectiveContext(
        family=operator.family,
        candidate_id=candidate.candidate_id,
        settings=dict(candidate.settings),
    )

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        return scalar(active_params, buffers, batch, context)

    finite_difference = finite_difference_hvp(
        scalar_function,
        params,
        vector,
    )
    errors = _layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        finite_difference,
    )

    measurements = {
        "directional_abs_diff": errors["max_abs_diff"],
        "directional_rel_diff": errors["max_rel_diff"],
    }

    symmetry_vector = _batch_tree(batch, "symmetry_vector")
    _require_min_probe_norm(vector, "hvp reference vector")
    _require_min_probe_norm(symmetry_vector, "symmetry_vector")
    anchor_symmetry = candidate_factory(
        anchor_candidate,
        batch,
        symmetry_vector,
    )()
    left = _layout_aware_tree_dot(
        candidate.settings,
        _runtime_vector(
            symmetry_vector,
            candidate.settings,
            candidate_output,
            parameter_surface,
        ),
        candidate_output,
    )
    right = _layout_aware_tree_dot(
        candidate.settings,
        _runtime_vector(
            vector,
            candidate.settings,
            anchor_symmetry,
            parameter_surface,
        ),
        anchor_symmetry,
    )
    measurements["symmetry_max_abs_diff"] = float((left - right).abs().item())

    return measurements


def _ggn_inner_product_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    anchor_candidate: Candidate,
    candidate_factory: OperationFactory,
    parameter_surface: ParameterSurface | None,
) -> dict[str, float]:
    if operator.kind != "ggnvp":
        return {}

    symmetry_vector = _batch_tree(batch, "symmetry_vector")
    _require_min_probe_norm(vector, "ggn reference vector")
    _require_min_probe_norm(symmetry_vector, "symmetry_vector")
    anchor_symmetry = candidate_factory(
        anchor_candidate,
        batch,
        symmetry_vector,
    )()
    left = _layout_aware_tree_dot(
        candidate.settings,
        _runtime_vector(
            symmetry_vector,
            candidate.settings,
            candidate_output,
            parameter_surface,
        ),
        candidate_output,
    )
    right = _layout_aware_tree_dot(
        candidate.settings,
        _runtime_vector(
            vector,
            candidate.settings,
            anchor_symmetry,
            parameter_surface,
        ),
        anchor_symmetry,
    )

    return {"inner_abs_diff": float((left - right).abs().item())}


def _require_min_probe_norm(vector: TensorTree, name: str) -> None:
    norm = float(_flatten_vector(vector).norm().detach().cpu())
    min_norm = STANDARD_THRESHOLDS["min_probe_norm"]

    if norm >= min_norm:
        return

    message = f"{name} norm is below min_probe_norm"
    raise ReferenceFailedError(message)


def _matrix_symmetry_error(matrix: torch.Tensor) -> float:
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "metric matrix")

    return float((matrix - matrix.T).abs().max().item())


def _matrix_psd_violation(matrix: torch.Tensor) -> float:
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "metric matrix")

    min_eigenvalue = torch.linalg.eigvalsh(matrix).min()

    return float(torch.clamp(-min_eigenvalue, min=0.0).item())


def _inverse_metric_matrix(
    operator: OperatorSpec,
    matrix: torch.Tensor,
    batch: Batch,
) -> torch.Tensor:
    if _inverse_metric_damping_kind(operator) == "per_group":
        return _per_group_damped_metric_matrix(operator, matrix, batch)

    return _damped_metric_matrix(matrix, _inverse_metric_damping(operator))


def _per_group_damped_metric_matrix(
    operator: OperatorSpec,
    matrix: torch.Tensor,
    batch: Batch,
) -> torch.Tensor:
    kind = _metric_representation_kind(operator)

    if kind == "block_diagonal":
        blocks = _metric_blocks(batch)
        damped = torch.block_diag(
            *starmap(
                _damped_metric_matrix,
                zip(blocks, _block_metric_dampings(operator, blocks), strict=True),
            )
        )
    elif kind == "kfac_factors":
        factor_batch = _kfac_factor_batch(batch)
        damped_blocks = []

        for block in _kfac_blocks(operator):
            left = _kfac_factor(factor_batch, block.left_factor_key)
            right = _kfac_factor(factor_batch, block.right_factor_key)
            dense_block = torch.kron(left, right)
            damping = _inverse_metric_group_damping(operator, block.parameter_name)
            damped_blocks.append(_damped_metric_matrix(dense_block, damping))

        damped = torch.block_diag(*damped_blocks)
    else:
        message = f"per_group damping is not lowered for metric kind: {kind}"
        raise MaterializationError(message)

    if damped.shape != matrix.shape:
        message = "per_group damping matrix does not match metric matrix shape"
        raise MaterializationError(message)

    _require_finite_tensor(damped, "per-group damped metric matrix")

    return damped


def _metric_reference_output(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    matrix = _metric_dense_matrix(operator, batch, vector)
    flat_vector = _flatten_vector(vector)
    _require_finite_tensor(matrix, "metric matrix")
    _require_finite_tensor(flat_vector, "metric vector")

    if operator.kind == "metric":
        result = dense_metric_multiply(matrix, flat_vector)
    elif operator.kind == "inverse_metric":
        result = dense_metric_inverse_multiply(
            _inverse_metric_matrix(operator, matrix, batch),
            flat_vector,
        )
    else:
        message = f"metric reference output is not supported for {operator.kind}"
        raise MaterializationError(message)

    _require_finite_tensor(result, "metric reference result")

    return _wrap_flat_vector(vector, result)


def _metric_dense_matrix(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> torch.Tensor:
    kind = _metric_representation_kind(operator)

    if kind == "dense_matrix":
        matrix = _batch_tensor(batch, "metric_matrix")
    elif kind == "diagonal_tree":
        diagonal = _flatten_vector(_metric_diagonal_tree(batch, vector))
        matrix = torch.diag(diagonal)
    elif kind == "block_diagonal":
        matrix = torch.block_diag(*_metric_blocks(batch))
    elif kind == "kfac_factors":
        matrix = _kfac_dense_matrix(operator, batch)
    elif kind == "ekfac_factors":
        matrix = _ekfac_dense_matrix(batch, vector)
    elif kind == "low_rank_factors":
        basis, diagonal = _low_rank_factors(batch, vector)
        matrix = basis @ basis.T + torch.diag(diagonal)
    elif kind == "ggn_derived_factors":
        jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
        matrix = jacobian.T @ loss_hessian @ jacobian
    else:
        message = f"metric representation has no dense reference: {kind}"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "metric matrix")

    return matrix


def _metric_diagonal_tree(batch: Batch, vector: TensorTree) -> TensorTree:
    diagonal = tree_map2(
        lambda diag, template: diag.reshape_as(template),
        _batch_tree(batch, "metric_diagonal"),
        vector,
    )
    _require_finite_tree(diagonal, "metric diagonal")

    return diagonal


def _diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    diagonal = _metric_diagonal_tree(batch, vector)
    result = _tree_elementwise_mul_runtime(settings, diagonal, vector)
    _require_finite_tree(result, "metric result")

    return result


def _diagonal_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    damping = _inverse_metric_damping(operator)
    diagonal = _metric_diagonal_tree(batch, vector)
    denominator = _tree_add_scalar_runtime(settings, diagonal, damping)
    result = _tree_elementwise_div_runtime(settings, vector, denominator)
    _require_finite_tree(result, "inverse metric result")

    return result


def _diagonal_inverse_metric_multiply_batch(
    execution: "StandardExecution",
) -> TensorTree:
    damping = _inverse_metric_damping(execution.operator)
    diagonal = _flatten_vector(_metric_diagonal_tree(execution.batch, execution.params))
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    result = vector_batch / (diagonal + damping)
    _require_finite_tensor(result, "batched diagonal inverse metric result")

    return _wrap_flat_vector_batch(execution.params, result)


def _metric_blocks(batch: Batch) -> tuple[torch.Tensor, ...]:
    value = batch.get("metric_blocks")

    if not isinstance(value, tuple) or not value:
        message = "metric blocks are missing"
        raise MaterializationError(message)

    for block in value:
        if not isinstance(block, torch.Tensor):
            message = "metric block must be a tensor"
            raise MaterializationError(message)

        if block.ndim != MATRIX_DIMS or block.shape[0] != block.shape[1]:
            message = "metric block must be square"
            raise MaterializationError(message)

        _require_finite_tensor(block, "metric block")

    return value


def _block_diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    result = _block_diagonal_apply(
        _metric_blocks(batch),
        _flatten_vector(vector),
        settings,
        "metric block multiply",
    )

    return _wrap_flat_vector(vector, result)


def _block_diagonal_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    blocks = _metric_blocks(batch)
    result = _block_diagonal_solve(
        blocks,
        _flatten_vector(vector),
        _block_metric_dampings(operator, blocks),
    )

    return _wrap_flat_vector(vector, result)


def _block_diagonal_inverse_metric_multiply_batch(
    execution: "StandardExecution",
) -> TensorTree:
    blocks = _metric_blocks(execution.batch)
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    result = _block_diagonal_solve_batch(
        blocks,
        vector_batch,
        _block_metric_dampings(execution.operator, blocks),
    )

    return _wrap_flat_vector_batch(execution.params, result)


def _block_metric_dampings(
    operator: OperatorSpec,
    blocks: tuple[torch.Tensor, ...],
) -> tuple[float, ...]:
    if _inverse_metric_damping_kind(operator) == "per_group":
        names = _block_diagonal_group_names(operator)

        if len(names) != len(blocks):
            message = "per_group damping keys do not match metric block count"
            raise MaterializationError(message)

        return tuple(_inverse_metric_group_damping(operator, name) for name in names)

    damping = _inverse_metric_damping(operator)

    return tuple(damping for _ in blocks)


def _block_diagonal_group_names(operator: OperatorSpec) -> tuple[str, ...]:
    representation = _metric_representation(operator)
    value = representation.get("block_names")

    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(name, str) for name in value)
    ):
        message = "block-diagonal per_group damping requires named metric blocks"
        raise MaterializationError(message)

    return value


def _block_diagonal_apply(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    settings: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    parts = []
    offset = 0

    for block in blocks:
        width = block.shape[1]
        part = vector[offset : offset + width]

        if part.numel() != width:
            message = "metric blocks do not match vector length"
            raise MaterializationError(message)

        parts.append(_matmul_runtime(settings, block, part))
        offset += width

    if offset != vector.numel():
        message = "metric blocks do not match vector length"
        raise MaterializationError(message)

    result = torch.cat(tuple(parts))
    _require_finite_tensor(result, name)

    return result


def _block_diagonal_solve(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    dampings: tuple[float, ...],
) -> torch.Tensor:
    result = _block_diagonal_solve_batch(
        blocks,
        vector.unsqueeze(0),
        dampings,
    )[0]
    _require_finite_tensor(result, "inverse metric block solve")

    return result


def _block_diagonal_solve_batch(
    blocks: tuple[torch.Tensor, ...],
    vector_batch: torch.Tensor,
    dampings: tuple[float, ...],
) -> torch.Tensor:
    if vector_batch.ndim != MATRIX_DIMS:
        message = "batched block inverse vectors must flatten to a matrix"
        raise MaterializationError(message)

    parts = []
    offset = 0

    for block, damping in zip(blocks, dampings, strict=True):
        width = block.shape[1]
        part = vector_batch[:, offset : offset + width]

        if part.shape[1] != width:
            message = "metric blocks do not match vector width"
            raise MaterializationError(message)

        solved = torch.linalg.solve(
            _damped_metric_matrix(block, damping),
            part.T,
        ).T
        parts.append(solved)
        offset += width

    if offset != vector_batch.shape[1]:
        message = "metric blocks do not match vector width"
        raise MaterializationError(message)

    result = torch.cat(tuple(parts), dim=1)
    _require_finite_tensor(result, "batched inverse metric block solve")

    return result


def _low_rank_factors(
    batch: Batch,
    vector: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = batch.get("low_rank_factors")

    if not isinstance(value, Mapping):
        message = "low_rank_factors must be a mapping"
        raise MaterializationError(message)

    basis = value.get("basis")
    diagonal = value.get("diagonal")
    width = _flatten_vector(vector).numel()

    if not isinstance(basis, torch.Tensor) or basis.ndim != MATRIX_DIMS:
        message = "low_rank_factors.basis must be a two-dimensional tensor"
        raise MaterializationError(message)

    if basis.shape[0] != width:
        message = "low_rank_factors.basis row count must match vector length"
        raise MaterializationError(message)

    if not isinstance(diagonal, torch.Tensor) or diagonal.ndim != 1:
        message = "low_rank_factors.diagonal must be a one-dimensional tensor"
        raise MaterializationError(message)

    if diagonal.numel() != width:
        message = "low_rank_factors.diagonal length must match vector length"
        raise MaterializationError(message)

    _require_finite_tensor(basis, "low-rank basis")
    _require_finite_tensor(diagonal, "low-rank diagonal")

    return basis, diagonal


def _low_rank_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = _runtime_intermediate_tensor(_flatten_vector(vector), settings)
    basis, diagonal = _low_rank_factors(batch, vector)
    basis = _runtime_intermediate_tensor(basis, settings)
    diagonal = _runtime_intermediate_tensor(diagonal, settings)
    basis_projection = _matmul_runtime(settings, basis.T, flat_vector)
    result = (
        _accumulation_tensor(diagonal, settings)
        * _accumulation_tensor(flat_vector, settings)
    ) + _matmul_runtime(settings, basis, basis_projection)
    _require_finite_tensor(result, "low-rank metric result")

    return _wrap_flat_vector(vector, result)


def _low_rank_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    result = _low_rank_inverse_metric_flat_batch(
        operator,
        batch,
        vector,
        _flatten_vector(vector).unsqueeze(0),
    )[0]
    _require_finite_tensor(result, "low-rank inverse metric result")

    return _wrap_flat_vector(vector, result)


def _low_rank_inverse_metric_flat_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    vector_batch: torch.Tensor,
) -> torch.Tensor:
    basis, diagonal = _low_rank_factors(batch, template)
    base_diagonal = diagonal + _inverse_metric_damping(operator)
    inverse_base_vectors = vector_batch / base_diagonal
    inverse_base_basis = basis / base_diagonal.unsqueeze(1)
    inner = (
        torch.eye(
            basis.shape[1],
            dtype=basis.dtype,
            device=basis.device,
        )
        + basis.T @ inverse_base_basis
    )
    correction = (
        inverse_base_basis @ torch.linalg.solve(inner, basis.T @ inverse_base_vectors.T)
    ).T
    result = inverse_base_vectors - correction
    _require_finite_tensor(result, "batched low-rank inverse metric result")

    return result


def _low_rank_inverse_metric_multiply_batch(
    execution: "StandardExecution",
) -> TensorTree:
    result = _low_rank_inverse_metric_flat_batch(
        execution.operator,
        execution.batch,
        execution.params,
        _flat_inverse_metric_vector_batch(execution),
    )

    return _wrap_flat_vector_batch(execution.params, result)


def _kfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    metric = KFACMetricOperator(_kfac_blocks(operator), settings=settings)

    return metric.multiply(_kfac_factor_batch(batch), vector)


def _kfac_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    metric = KFACMetricOperator(
        _kfac_blocks(operator),
        damping=_inverse_metric_damping_payload(operator),
        damping_kind=_inverse_metric_damping_kind(operator),
        damping_policy=_inverse_metric_damping_policy(operator),
    )

    return metric.inverse_multiply(_kfac_factor_batch(batch), vector)


def _kfac_inverse_metric_multiply_batch(execution: "StandardExecution") -> TensorTree:
    batch = _kfac_factor_batch(execution.batch)
    vector_map = _kfac_vector_map(execution.vector)
    vector_in_dims = _vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )
    vector_count = _vector_tree_batch_size(execution.vector, vector_in_dims)
    result = {}

    if not isinstance(vector_in_dims, Mapping):
        message = "KFAC vectorization.in_dims must be a mapping"
        raise MaterializationError(message)

    for block in _kfac_blocks(execution.operator):
        left = _kfac_factor(batch, block.left_factor_key)
        right = _kfac_factor(batch, block.right_factor_key)
        value = _kfac_batched_vector_leaf(
            vector_map,
            vector_in_dims,
            block,
            left,
            right,
            vector_count,
        )
        product = _kfac_inverse_product_batch(
            execution.operator,
            block.parameter_name,
            left,
            right,
            value,
        )
        _require_finite_tensor(
            product,
            f"batched inverse KFAC metric result {block.parameter_name}",
        )
        result[block.parameter_name] = product

    return result


def _kfac_square_root_apply(
    execution: "StandardExecution",
    *,
    inverse: bool,
) -> TensorTree:
    batch = _kfac_factor_batch(execution.batch)
    vector_map = _kfac_vector_map(execution.vector)
    result = {}

    for block in _kfac_blocks(execution.operator):
        left = _kfac_factor(batch, block.left_factor_key)
        right = _kfac_factor(batch, block.right_factor_key)
        value = _kfac_vector_leaf(vector_map, block)
        _require_kfac_shapes(block, left, right, value)

        if inverse:
            product = _kfac_inverse_square_root_product(
                execution.operator,
                block.parameter_name,
                left,
                right,
                value,
            )
        else:
            product = _kfac_square_root_product(left, right, value)

        _require_finite_tensor(
            product,
            f"KFAC square-root metric result {block.parameter_name}",
        )
        result[block.parameter_name] = product

    return result


def _kfac_square_root_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    _require_positive_spectrum(left_eigenvalues, "KFAC left spectrum")
    _require_positive_spectrum(right_eigenvalues, "KFAC right spectrum")

    return _kfac_eigenbasis_scale(
        left_eigenvectors,
        right_eigenvectors,
        torch.sqrt(left_eigenvalues)[:, None] * torch.sqrt(right_eigenvalues)[None, :],
        value,
    )


def _kfac_inverse_square_root_product(
    operator: OperatorSpec,
    parameter_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    damping_kind = _inverse_metric_damping_kind(operator)
    damping = _resolved_group_damping(
        _inverse_metric_damping_payload(operator),
        damping_kind,
        parameter_name,
    )
    resolved_kind = _resolved_group_damping_kind(damping_kind)

    if resolved_kind == "scalar":
        spectrum = left_eigenvalues[:, None] * right_eigenvalues[None, :] + damping
        _require_positive_spectrum(spectrum.reshape(-1), "KFAC inverse spectrum")
        scale = torch.rsqrt(spectrum)
    elif resolved_kind == "kfac_pi":
        left_shift, right_shift = _kfac_pi_shifts(
            left,
            right,
            damping,
            _inverse_metric_damping_policy(operator),
        )
        left_spectrum = left_eigenvalues + left_shift
        right_spectrum = right_eigenvalues + right_shift
        _require_positive_spectrum(left_spectrum, "KFAC pi left spectrum")
        _require_positive_spectrum(right_spectrum, "KFAC pi right spectrum")
        scale = (
            torch.rsqrt(left_spectrum)[:, None] * torch.rsqrt(right_spectrum)[None, :]
        )
    else:
        message = (
            f"KFAC inverse square root does not lower damping kind: {damping_kind}"
        )
        raise MaterializationError(message)

    return _kfac_eigenbasis_scale(
        left_eigenvectors,
        right_eigenvectors,
        scale,
        value,
    )


def _kfac_eigenbasis_scale(
    left_eigenvectors: torch.Tensor,
    right_eigenvectors: torch.Tensor,
    scale: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    rotated = left_eigenvectors.T @ value @ right_eigenvectors
    scaled = scale * rotated

    return left_eigenvectors @ scaled @ right_eigenvectors.T


def _kfac_dense_matrix(operator: OperatorSpec, batch: Batch) -> torch.Tensor:
    factor_batch = _kfac_factor_batch(batch)
    dense_blocks = []

    for block in _kfac_blocks(operator):
        left = _kfac_factor(factor_batch, block.left_factor_key)
        right = _kfac_factor(factor_batch, block.right_factor_key)
        dense_blocks.append(torch.kron(left, right))

    matrix = torch.block_diag(*dense_blocks)
    _require_finite_tensor(matrix, "KFAC dense matrix")

    return matrix


def _kfac_blocks(operator: OperatorSpec) -> tuple["KFACMetricBlock", ...]:
    representation = _metric_representation(operator)
    raw_blocks = representation.get("blocks")

    if not isinstance(raw_blocks, tuple) or not raw_blocks:
        message = "kfac_factors representation requires non-empty blocks"
        raise MaterializationError(message)

    blocks = []

    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            message = "KFAC block descriptor must be a mapping"
            raise MaterializationError(message)

        parameter_name = raw_block.get("parameter")
        left_factor_key = raw_block.get("left_factor")
        right_factor_key = raw_block.get("right_factor")

        if not isinstance(parameter_name, str):
            message = "KFAC block parameter must be a string"
            raise MaterializationError(message)

        if not isinstance(left_factor_key, str):
            message = "KFAC block left_factor must be a string"
            raise MaterializationError(message)

        if not isinstance(right_factor_key, str):
            message = "KFAC block right_factor must be a string"
            raise MaterializationError(message)

        blocks.append(
            KFACMetricBlock(
                parameter_name,
                left_factor_key,
                right_factor_key,
            )
        )

    return tuple(blocks)


def _kfac_factor_batch(batch: Batch) -> Batch:
    value = batch.get("kfac_factors")

    if not isinstance(value, Mapping):
        message = "kfac_factors must be a mapping"
        raise MaterializationError(message)

    return value


def _ekfac_dense_matrix(batch: Batch, vector: TensorTree) -> torch.Tensor:
    blocks = []

    for key, value in _ekfac_vector_map(vector).items():
        eigvecs_a, eigvecs_g, eigenvalues = _ekfac_factors(batch, key, value)
        basis = torch.kron(eigvecs_a, eigvecs_g)
        block = basis @ torch.diag(eigenvalues.reshape(-1)) @ basis.T
        blocks.append(block)

    matrix = torch.block_diag(*blocks)
    _require_finite_tensor(matrix, "EKFAC dense matrix")

    return matrix


def _ekfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator

    return _ekfac_apply(batch, vector, settings, inverse=False, square_root=False)


def _ekfac_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings

    return _ekfac_apply(
        batch,
        vector,
        {},
        inverse=True,
        square_root=False,
        damping=_inverse_metric_damping(operator),
    )


def _ekfac_inverse_metric_multiply_batch(execution: "StandardExecution") -> TensorTree:
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    matrix = _ekfac_dense_matrix(execution.batch, execution.params)
    result = torch.linalg.solve(
        _damped_metric_matrix(matrix, _inverse_metric_damping(execution.operator)),
        vector_batch.T,
    ).T
    _require_finite_tensor(result, "batched inverse EKFAC metric result")

    return _wrap_flat_vector_batch(execution.params, result)


def _ekfac_square_root_apply(
    execution: "StandardExecution",
    *,
    inverse: bool,
) -> TensorTree:
    return _ekfac_apply(
        execution.batch,
        execution.vector,
        execution.candidate.settings,
        inverse=inverse,
        square_root=True,
        damping=_inverse_metric_damping(execution.operator) if inverse else 0.0,
    )


def _ekfac_apply(
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    inverse: bool,
    square_root: bool,
    damping: float = 0.0,
) -> TensorTree:
    result = {}

    for key, value in _ekfac_vector_map(vector).items():
        eigvecs_a, eigvecs_g, eigenvalues = _ekfac_factors(batch, key, value)
        result[key] = _ekfac_apply_leaf(
            eigvecs_a,
            eigvecs_g,
            eigenvalues,
            value,
            settings,
            inverse=inverse,
            square_root=square_root,
            damping=damping,
        )

    return result


def _ekfac_apply_leaf(
    eigvecs_a: torch.Tensor,
    eigvecs_g: torch.Tensor,
    eigenvalues: torch.Tensor,
    value: torch.Tensor,
    settings: Mapping[str, Any],
    *,
    inverse: bool,
    square_root: bool,
    damping: float,
) -> torch.Tensor:
    spectrum = eigenvalues + damping if inverse else eigenvalues
    _require_positive_spectrum(spectrum.reshape(-1), "EKFAC spectrum")
    rotated = _matmul_runtime(
        settings,
        _matmul_runtime(settings, eigvecs_a.T, value),
        eigvecs_g,
    )

    if square_root:
        factors = torch.rsqrt(spectrum) if inverse else torch.sqrt(spectrum)
    elif inverse:
        factors = torch.reciprocal(spectrum)
    else:
        factors = spectrum

    scaled = factors * rotated
    result = _matmul_runtime(
        settings,
        _matmul_runtime(settings, eigvecs_a, scaled),
        eigvecs_g.T,
    )
    _require_finite_tensor(result, "EKFAC metric result")

    return result


def _ekfac_vector_map(vector: TensorTree) -> dict[str, torch.Tensor]:
    if type(vector) is not dict:
        message = "EKFAC vector must be a tensor-tree mapping"
        raise MaterializationError(message)

    result = {}

    for key, value in vector.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            message = "EKFAC vector leaves must be named tensors"
            raise MaterializationError(message)

        if value.ndim != MATRIX_DIMS:
            message = f"EKFAC vector leaf must be a matrix: {key}"
            raise MaterializationError(message)

        _require_finite_tensor(value, f"EKFAC vector {key}")
        result[key] = value

    return result


def _ekfac_factor_map(batch: Batch, key: str) -> Mapping[str, torch.Tensor]:
    value = batch.get(key)

    if not isinstance(value, Mapping):
        message = f"{key} must be a mapping"
        raise MaterializationError(message)

    return value


def _ekfac_factors(
    batch: Batch,
    key: str,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eigvecs_a = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_eigvecs_a"),
        key,
        "ekfac_eigvecs_a",
    )
    eigvecs_g = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_eigvecs_g"),
        key,
        "ekfac_eigvecs_g",
    )
    eigenvalues = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_corrected_eigenvalues"),
        key,
        "ekfac_corrected_eigenvalues",
    )

    if eigvecs_a.ndim != MATRIX_DIMS or eigvecs_a.shape[0] != eigvecs_a.shape[1]:
        message = f"EKFAC eigvecs_a must be square for {key}"
        raise MaterializationError(message)

    if eigvecs_g.ndim != MATRIX_DIMS or eigvecs_g.shape[0] != eigvecs_g.shape[1]:
        message = f"EKFAC eigvecs_g must be square for {key}"
        raise MaterializationError(message)

    if tuple(value.shape) != (eigvecs_a.shape[0], eigvecs_g.shape[0]):
        message = f"EKFAC vector leaf shape mismatch for {key}"
        raise MaterializationError(message)

    if tuple(eigenvalues.shape) != tuple(value.shape):
        message = f"EKFAC corrected eigenvalues shape mismatch for {key}"
        raise MaterializationError(message)

    return eigvecs_a, eigvecs_g, eigenvalues


def _ekfac_factor_tensor(
    values: Mapping[str, torch.Tensor],
    key: str,
    label: str,
) -> torch.Tensor:
    tensor = values.get(key)

    if not isinstance(tensor, torch.Tensor):
        message = f"{label} is missing or not a tensor: {key}"
        raise MaterializationError(message)

    _require_finite_tensor(tensor, f"EKFAC factor {label}.{key}")

    return tensor


def _ggn_metric_factors(
    batch: Batch,
    vector: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = batch.get("ggn_factors")

    if not isinstance(value, Mapping):
        message = "ggn_factors must be a mapping"
        raise MaterializationError(message)

    jacobian = value.get("jacobian")
    loss_hessian = value.get("loss_hessian")
    width = _flatten_vector(vector).numel()

    if not isinstance(jacobian, torch.Tensor) or jacobian.ndim != MATRIX_DIMS:
        message = "ggn_factors.jacobian must be a two-dimensional tensor"
        raise MaterializationError(message)

    if jacobian.shape[1] != width:
        message = "ggn_factors.jacobian column count must match vector length"
        raise MaterializationError(message)

    if (
        not isinstance(loss_hessian, torch.Tensor)
        or loss_hessian.ndim != MATRIX_DIMS
        or loss_hessian.shape[0] != loss_hessian.shape[1]
    ):
        message = "ggn_factors.loss_hessian must be a square matrix"
        raise MaterializationError(message)

    if loss_hessian.shape[0] != jacobian.shape[0]:
        message = "ggn_factors.loss_hessian shape must match jacobian rows"
        raise MaterializationError(message)

    _require_finite_tensor(jacobian, "GGN metric jacobian")
    _require_finite_tensor(loss_hessian, "GGN metric loss hessian")

    return jacobian, loss_hessian


def _ggn_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = _flatten_vector(vector)
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    output_vector = _matmul_runtime(settings, jacobian, flat_vector)
    loss_vector = _matmul_runtime(settings, loss_hessian, output_vector)
    result = _matmul_runtime(settings, jacobian.T, loss_vector)
    _require_finite_tensor(result, "GGN-derived metric result")

    return _wrap_flat_vector(vector, result)


def _ggn_derived_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    flat_vector = _flatten_vector(vector)
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    result = _ggn_derived_inverse_metric_flat_rhs(
        operator,
        jacobian,
        loss_hessian,
        flat_vector[:, None],
    ).squeeze(1)
    _require_finite_tensor(result, "GGN-derived inverse metric result")

    return _wrap_flat_vector(vector, result)


def _ggn_derived_inverse_metric_multiply_batch(
    execution: "StandardExecution",
) -> TensorTree:
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    jacobian, loss_hessian = _ggn_metric_factors(execution.batch, execution.params)
    result = _ggn_derived_inverse_metric_flat_rhs(
        execution.operator,
        jacobian,
        loss_hessian,
        vector_batch.T,
    ).T
    _require_finite_tensor(result, "batched GGN-derived inverse metric result")

    return _wrap_flat_vector_batch(execution.params, result)


def _ggn_derived_inverse_metric_flat_rhs(
    operator: OperatorSpec,
    jacobian: torch.Tensor,
    loss_hessian: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    damping = _inverse_metric_damping(operator)

    if damping <= 0.0:
        message = "GGN-derived factorized inverse requires positive damping"
        raise MaterializationError(message)

    if rhs.ndim != MATRIX_DIMS or rhs.shape[0] != jacobian.shape[1]:
        message = "GGN-derived inverse RHS shape must match parameter width"
        raise MaterializationError(message)

    eigenvalues, eigenvectors = torch.linalg.eigh(loss_hessian)
    _require_nonnegative_spectrum(eigenvalues, "GGN-derived loss Hessian")
    sqrt_loss_hessian = (
        eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    factor = sqrt_loss_hessian @ jacobian
    capacitance = (
        torch.eye(
            factor.shape[0],
            dtype=factor.dtype,
            device=factor.device,
        )
        + (factor @ factor.T) / damping
    )
    correction = factor.T @ torch.linalg.solve(capacitance, factor @ rhs)
    result = rhs / damping - correction / (damping * damping)
    _require_finite_tensor(result, "GGN-derived inverse metric flat result")

    return result


def _damped_metric_matrix(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    if damping <= 0.0:
        return matrix

    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)

    return matrix + damping * identity


def _inverse_metric_damping(operator: OperatorSpec) -> float:
    value = operator.semantics.get("damping")

    if not isinstance(value, float):
        message = "inverse metric damping must be a float"
        raise MaterializationError(message)

    if value < 0.0:
        message = "inverse metric damping must be nonnegative"
        raise MaterializationError(message)

    return value


def _inverse_metric_damping_values(operator: OperatorSpec) -> Mapping[str, float]:
    value = operator.semantics.get("damping")

    if not isinstance(value, Mapping) or not value:
        message = "inverse metric per_group damping must be a nonempty mapping"
        raise MaterializationError(message)

    result = {}

    for key, damping in value.items():
        if not isinstance(key, str):
            message = "inverse metric per_group damping keys must be strings"
            raise MaterializationError(message)

        if (
            not isinstance(damping, float | int)
            or isinstance(damping, bool)
            or damping < 0.0
        ):
            message = "inverse metric per_group damping values must be nonnegative"
            raise MaterializationError(message)

        result[key] = float(damping)

    return result


def _inverse_metric_group_damping(operator: OperatorSpec, group: str) -> float:
    values = _inverse_metric_damping_values(operator)
    value = values.get(group)

    if value is None:
        message = f"inverse metric per_group damping is missing group: {group}"
        raise MaterializationError(message)

    return value


def _inverse_metric_min_damping(operator: OperatorSpec) -> float:
    if _inverse_metric_damping_kind(operator) == "per_group":
        return min(_inverse_metric_damping_values(operator).values())

    return _inverse_metric_damping(operator)


def _inverse_metric_damping_payload(
    operator: OperatorSpec,
) -> float | Mapping[str, float]:
    if _inverse_metric_damping_kind(operator) == "per_group":
        return _inverse_metric_damping_values(operator)

    return _inverse_metric_damping(operator)


def _resolved_group_damping(
    damping: float | Mapping[str, float],
    damping_kind: str,
    group: str,
) -> float:
    if damping_kind == "per_group":
        values = _required_group_damping_mapping(damping)
        value = values.get(group)

        if value is None:
            message = f"per_group damping is missing group: {group}"
            raise MaterializationError(message)

        return value

    if not isinstance(damping, float):
        message = "inverse metric damping must be a float"
        raise MaterializationError(message)

    return damping


def _required_group_damping_mapping(
    damping: float | Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(damping, Mapping):
        message = "per_group damping payload must be a mapping"
        raise MaterializationError(message)

    result = {}

    for key, value in damping.items():
        if not isinstance(key, str):
            message = "per_group damping keys must be strings"
            raise MaterializationError(message)

        if not isinstance(value, float | int) or isinstance(value, bool):
            message = "per_group damping values must be numeric"
            raise MaterializationError(message)

        result[key] = float(value)

    return result


def _resolved_group_damping_kind(damping_kind: str) -> str:
    if damping_kind == "per_group":
        return "scalar"

    return damping_kind


def _inverse_metric_tolerance(operator: OperatorSpec) -> float | None:
    value = operator.semantics.get("tol")

    if value is None:
        return None

    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        message = "inverse metric tol must be positive and finite"
        raise MaterializationError(message)

    return value


def _inverse_metric_damping_kind(operator: OperatorSpec) -> str:
    value = operator.semantics.get("damping_kind")

    if not isinstance(value, str):
        message = "inverse metric damping_kind must be a string"
        raise MaterializationError(message)

    return value


def _inverse_metric_damping_policy(operator: OperatorSpec) -> str | None:
    value = operator.semantics.get("damping_policy")

    if value is None:
        return None

    if not isinstance(value, str):
        message = "inverse metric damping_policy must be a string"
        raise MaterializationError(message)

    return value


def _matrix_condition_number(matrix: torch.Tensor) -> float:
    _require_finite_tensor(matrix, "metric matrix")
    condition = torch.linalg.cond(matrix)

    return float(condition.item())


def _inverse_residual(
    matrix: torch.Tensor,
    inverse_result: torch.Tensor,
    vector: torch.Tensor,
) -> float:
    _require_finite_tensor(matrix, "metric matrix")
    _require_finite_tensor(inverse_result, "inverse result")
    _require_finite_tensor(vector, "metric vector")

    residual = (matrix @ inverse_result.reshape(-1) - vector.reshape(-1)).norm()
    denominator = vector.reshape(-1).norm()

    if math.isclose(float(denominator.item()), 0.0, rel_tol=0.0, abs_tol=0.0):
        return float(residual.item())

    return float((residual / denominator).item())


@contextlib.contextmanager
def deferred_runtime_finite_checks() -> Iterator[None]:
    """Skip runtime tensor scans inside a timed or compiled callable."""
    previous = FINITE_CHECKS_ENABLED[0]
    FINITE_CHECKS_ENABLED[0] = False

    try:
        yield
    finally:
        FINITE_CHECKS_ENABLED[0] = previous


def _call_with_deferred_finite_checks(
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    with deferred_runtime_finite_checks():
        return callback(*args)


@contextlib.contextmanager
def _disabled_backend_settings() -> Iterator[None]:
    previous = BACKEND_SETTINGS_ENABLED[0]
    BACKEND_SETTINGS_ENABLED[0] = False

    try:
        yield
    finally:
        BACKEND_SETTINGS_ENABLED[0] = previous


def _call_compiled_body(callback: Callable[..., Any], *args: Any) -> Any:
    with deferred_runtime_finite_checks(), _disabled_backend_settings():
        return callback(*args)


def _call_compiled_operation(
    settings: Mapping[str, Any],
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    return _run_with_backend_settings(
        settings,
        lambda: _call_compiled_body(callback, *args),
    )


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not FINITE_CHECKS_ENABLED[0]:
        return

    check_tensor = _finite_check_tensor(tensor)

    if not torch.isfinite(check_tensor).all().item():
        message = f"{name} contains nonfinite values"
        raise MaterializationError(message)


def _finite_check_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _is_fp8_tensor(tensor):
        return tensor.to(torch.float32)

    return tensor


def _is_fp8_tensor(tensor: torch.Tensor) -> bool:
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    ):
        dtype = getattr(torch, dtype_name, None)

        if dtype is not None and tensor.dtype == dtype:
            return True

    return False


def _require_finite_tree(tree: TensorTree, name: str) -> None:
    for leaf in tree_leaves(tree):
        _require_finite_tensor(leaf, name)


def composition_runtime_config(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    children: Sequence[CompositionChild] = (),
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    component_signature: Mapping[str, Any],
    anchor_component_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Return runtime config for sequential operator composition."""
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    operation_factory = composition_operation_factory(
        operator,
        components=components,
        fused_components=fused_components,
        children=children,
    )
    reference_check = composition_reference_check(
        operator,
        components=components,
        anchor_components=anchor_components,
        fused_components=fused_components,
        thresholds=thresholds,
        numeric_bound_fields=bound_fields,
        children=children,
    )
    fused_component_keys = (
        ()
        if fused_components is None
        else tuple(tuple(key) for key in sorted(fused_components))
    )
    runtime_signature = {
        "runtime": "composition",
        "operator": operator.signature(),
        "order": _operator_composition_order(operator),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": bound_fields,
        "components": {
            "candidate": dict(component_signature),
            "anchor": dict(anchor_component_signature),
        },
        "fused_components": fused_component_keys,
        "children": tuple(child.name for child in children),
    }
    operation_factory = CallableOperationFactory(
        "vptune.composition_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.composition_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.composition_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.composition_reference_check"},
        reference_check,
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=_standard_materializer(operation_factory),
        axis_registry=axis_registry,
        reference_check_name="composition_anchor",
        signature=runtime_signature,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StandardExecution:
    """Inputs for one standard operator execution."""

    operator: OperatorSpec
    candidate: Candidate
    path: str
    batch: Batch
    vector: TensorTree
    params: ParameterTree
    buffers: BufferTree
    parameter_surface: ParameterSurface | None
    context: ObjectiveContext
    scalar_objectives: Mapping[str, ScalarObjective]
    function_objectives: Mapping[str, FunctionObjective]
    module: torch.nn.Module | None = None
    module_call: ModuleCallSpec | None = None
    teacher_objective: FunctionObjective | None = None
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None
    fusion_rewriter: Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None = (
        None
    )
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None
    activation_pack_hooks: ActivationPackHooks = dataclasses.field(default_factory=dict)
    activation_unpack_hooks: ActivationUnpackHooks = dataclasses.field(
        default_factory=dict
    )
    checkpoint_contexts: CheckpointContextFns = dataclasses.field(default_factory=dict)
    intermediate_transform: IntermediateTransform | None = None
    flat_parameter_vector: torch.Tensor | None = None
    flat_parameter_vector_batch: torch.Tensor | None = None
    compiled_inner: CandidateOperation | None = None
    compiled_model_forward: Callable[[Batch], object] | None = None
    compiled_scalar_function: Callable[[ParameterTree], torch.Tensor] | None = None
    compiled_score_matrix: Callable[[], torch.Tensor] | None = None
    compiled_ggn_jvp: Callable[[], tuple[TensorTree, TensorTree]] | None = None
    compiled_ggn_loss_product: Callable[[TensorTree, TensorTree], TensorTree] | None = (
        None
    )
    compiled_ggn_vjp: Callable[[TensorTree], TensorTree] | None = None
    prepared_gradient: CandidateOperation | None = None
    linearized_jvp: Callable[[TensorTree], TensorTree] | None = None
    linearized_hvp: Callable[[TensorTree], TensorTree] | None = None
    vjp_closure: Callable[[TensorTree], TensorTree] | None = None


PARAMETER_VECTOR_CACHE_OPERATOR_KINDS = (
    "hvp",
    "ggnvp",
    "fisher_vp",
    "sampled_fisher_vp",
    "empirical_fisher_vp",
    "metric",
    "sqrt_metric",
    "inverse_sqrt_metric",
    "inverse_metric",
)


def _execution_with_vector(
    execution: StandardExecution,
    vector: TensorTree,
    **changes: Any,
) -> StandardExecution:
    return dataclasses.replace(
        execution,
        vector=vector,
        flat_parameter_vector=None,
        flat_parameter_vector_batch=None,
        **changes,
    )


def standard_operation_factory(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    parameter_surface: ParameterSurface | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: ActivationPackHooks | None = None,
    activation_unpack_hooks: ActivationUnpackHooks | None = None,
    checkpoint_contexts: CheckpointContextFns | None = None,
    intermediate_transform: IntermediateTransform | None = None,
) -> OperationFactory:
    """Return an operation factory for package-owned standard operators."""
    scalar_map = {} if scalar_objectives is None else dict(scalar_objectives)
    function_map = {} if function_objectives is None else dict(function_objectives)
    pack_hook_map = {} if activation_pack_hooks is None else dict(activation_pack_hooks)
    unpack_hook_map = (
        {} if activation_unpack_hooks is None else dict(activation_unpack_hooks)
    )
    checkpoint_context_map = (
        {} if checkpoint_contexts is None else dict(checkpoint_contexts)
    )
    mmap_residency_callback = mmap_residency

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        _require_candidate_family(operator, candidate)
        _require_supported_standard_settings(
            operator,
            candidate,
            parameter_surface,
            mmap_residency_callback,
            fusion_rewriter,
            batch_layout,
            lm_head_chunker,
            pack_hook_map,
            unpack_hook_map,
            checkpoint_context_map,
        )
        _require_recomputed_teacher_objective(candidate.settings, teacher_objective)
        runtime_module = _runtime_fusion_module(module, candidate, fusion_rewriter)
        path = _runtime_path(operator, candidate)
        transformed_batch = _runtime_declared_batch_transforms(
            batch,
            candidate,
            batch_layout,
            lm_head_chunker,
        )
        _require_batch_inputs(operator, candidate, transformed_batch, phase="operation")
        runtime_params = _runtime_params(params, candidate.settings, parameter_surface)
        runtime_buffers = _runtime_buffers(buffers, candidate.settings)
        runtime_batch = _runtime_batch(
            transformed_batch,
            candidate.settings,
            move_input_residency=_move_input_residency_outside_measured_call(
                candidate.settings
            ),
            mmap_residency=mmap_residency_callback,
        )
        runtime_vector = _runtime_vector(
            vector,
            candidate.settings,
            runtime_params,
            parameter_surface,
            mmap_residency=mmap_residency_callback,
        )
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )
        execution = StandardExecution(
            operator=operator,
            candidate=candidate,
            path=path,
            batch=runtime_batch,
            vector=runtime_vector,
            params=runtime_params,
            buffers=runtime_buffers,
            parameter_surface=parameter_surface,
            context=context,
            scalar_objectives=scalar_map,
            function_objectives=function_map,
            module=runtime_module,
            module_call=module_call,
            teacher_objective=teacher_objective,
            batch_layout=batch_layout,
            lm_head_chunker=lm_head_chunker,
            mmap_residency=mmap_residency_callback,
            manual_recompute=manual_recompute,
            activation_pack_hooks=pack_hook_map,
            activation_unpack_hooks=unpack_hook_map,
            checkpoint_contexts=checkpoint_context_map,
            intermediate_transform=intermediate_transform,
        )
        _require_stateful_module_execution(execution)
        execution = _loss_scaled_execution(execution)
        _require_finite_execution_inputs(execution)
        execution = _prepare_standard_execution(execution)
        execution = _prepare_flat_vector_execution(execution)
        execution = _prepare_compile_boundary_execution(execution)
        output_buffer = _standard_output_buffer(execution)

        def operation() -> TensorTree:
            operation_execution = _execution_with_inside_input_residency(execution)

            return _run_with_backend_settings(
                candidate.settings,
                lambda: _run_with_call_grad_mode(
                    candidate.settings,
                    lambda: _runtime_output_to_buffer(
                        _runtime_output(
                            _run_with_buffer_mutation_check(
                                operation_execution,
                                lambda: _run_standard_operation(operation_execution),
                            ),
                            candidate.settings,
                            parameter_surface,
                        ),
                        output_buffer,
                    ),
                ),
            )

        activated_operation = _activation_operation(execution, operation)

        return _compile_operation(
            operator,
            candidate.settings,
            activated_operation,
        )

    return factory


def _prepare_standard_execution(execution: StandardExecution) -> StandardExecution:
    if execution.operator.kind == "gradient":
        return _prepare_gradient_execution(execution)

    if execution.operator.kind == "jvp":
        return _prepare_jvp_execution(execution)

    if execution.operator.kind == "vjp":
        return _prepare_vjp_execution(execution)

    if execution.operator.kind == "hvp":
        return _prepare_hvp_execution(execution)

    return execution


def _prepare_compile_boundary_execution(
    execution: StandardExecution,
) -> StandardExecution:
    settings = execution.candidate.settings
    boundary = settings.get("compile.boundary")

    if settings.get("compile.enabled") != "true" or not isinstance(boundary, str):
        return execution

    return _prepare_enabled_compile_boundary_execution(execution, settings, boundary)


def _prepare_flat_vector_execution(
    execution: StandardExecution,
) -> StandardExecution:
    if execution.operator.kind not in PARAMETER_VECTOR_CACHE_OPERATOR_KINDS:
        return execution

    if _uses_rectangular_square_root_input(execution):
        if "vectorization.mode" in execution.candidate.settings:
            message = (
                "rectangular closed-form square-root does not support vectorization"
            )
            raise MaterializationError(message)

        return execution

    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "vmap":
        return dataclasses.replace(
            execution,
            flat_parameter_vector_batch=_build_flat_vector_batch(execution),
        )

    if mode in {"single_loop", "manual_batch"}:
        return execution

    return dataclasses.replace(
        execution,
        flat_parameter_vector=_build_parameter_order_vector(execution),
    )


def _uses_rectangular_square_root_input(execution: StandardExecution) -> bool:
    if execution.operator.kind != "sqrt_metric":
        return False

    if execution.path != SQRT_METRIC_CLOSED_FORM_PATH:
        return False

    return _metric_representation_kind(execution.operator) in {
        "low_rank_factors",
        "ggn_derived_factors",
    }


def _prepare_enabled_compile_boundary_execution(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    boundary: str,
) -> StandardExecution:
    special_builder = {
        "model_forward": lambda: _prepare_model_forward_compile_boundary(
            execution,
            settings,
        ),
        "loss_closure": lambda: _prepare_loss_closure_compile_boundary(
            execution,
            settings,
        ),
    }.get(boundary)

    if special_builder is not None:
        return special_builder()

    inner_builders = {
        ("gradient", "gradient_closure"): lambda: _run_gradient_by_path(execution),
        ("jvp", "jvp_closure"): lambda: _run_jvp_by_path(execution),
        ("vjp", "vjp_closure"): lambda: _run_vjp_by_path(execution),
        ("hvp", "hvp_single_vector"): lambda: _run_hvp_single_vector(execution),
        ("hvp", "hvp_batched_vectors"): lambda: _run_hvp_by_path(execution),
        ("ggnvp", "ggn_full_product"): lambda: _run_ggnvp_by_path(execution),
        ("metric", "metric_multiply"): lambda: _metric_multiply_by_path(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.path,
            settings,
        ),
        ("inverse_metric", "inverse_metric_solve"): lambda: _run_inverse_metric_by_mode(
            execution
        ),
        (
            "sqrt_metric",
            "metric_sqrt_multiply",
        ): lambda: _metric_square_root_apply(
            execution,
            inverse=False,
            adjoint=False,
        ),
        (
            "inverse_sqrt_metric",
            "metric_sqrt_multiply",
        ): lambda: _metric_square_root_apply(
            execution,
            inverse=True,
            adjoint=False,
        ),
        ("metric_inner", "metric_inner_reduce"): lambda: _run_metric_inner(execution),
        (
            "inverse_metric_inner",
            "inverse_metric_inner_reduce",
        ): lambda: _run_inverse_metric_inner(execution),
    }
    score_builders = {
        ("fisher_vp", "fisher_score_grad"): lambda: (
            _score_gradient_matrix_from_builders(
                execution,
                "fisher_score_grad",
                FISHER_SCORE_GRADIENT_BUILDERS,
                "fisher score-gradient boundary requires a score-gradient path",
                use_compiled=False,
            )
        ),
        (
            "sampled_fisher_vp",
            "sampled_fisher_score_grad",
        ): lambda: _score_gradient_matrix_from_builders(
            execution,
            "sampled_fisher_score_grad",
            SAMPLED_FISHER_SCORE_GRADIENT_BUILDERS,
            "sampled Fisher score-gradient boundary requires a score-gradient path",
            use_compiled=False,
        ),
        (
            "empirical_fisher_vp",
            "empirical_fisher_example_grad",
        ): lambda: _score_gradient_matrix_from_builders(
            execution,
            "empirical_fisher_example_grad",
            EMPIRICAL_FISHER_GRADIENT_BUILDERS,
            "empirical Fisher example-gradient boundary requires a gradient path",
            use_compiled=False,
        ),
        (
            "per_example_gradient",
            "per_example_gradient",
        ): lambda: _per_example_gradient_matrix_without_manual_batch(execution),
    }
    builder = inner_builders.get((execution.operator.kind, boundary))

    if builder is not None:
        return _prepare_inner_compile_boundary(execution, settings, builder)

    score_builder = score_builders.get((execution.operator.kind, boundary))

    if score_builder is not None:
        return _prepare_score_matrix_compile_boundary(
            execution,
            settings,
            score_builder,
        )

    ggn_execution = _prepare_ggn_compile_boundary_execution(
        execution,
        settings,
        boundary,
    )

    if ggn_execution is not None:
        return ggn_execution

    return execution


def _prepare_ggn_compile_boundary_execution(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    boundary: str,
) -> StandardExecution | None:
    if execution.operator.kind != "ggnvp":
        return None

    if boundary == "ggn_loss_hessian_product":
        return _prepare_ggn_loss_product_compile_boundary(
            execution,
            settings,
            lambda output, output_jvp: _ggn_loss_hessian_product_by_path(
                execution,
                output,
                output_jvp,
            ),
        )

    if boundary == "ggn_jvp":
        return _prepare_ggn_jvp_compile_boundary(
            execution,
            settings,
            lambda: _ggn_output_and_jvp_by_path(
                execution,
                _ggn_tensor_function(execution),
            ),
        )

    if boundary == "ggn_vjp":
        return _prepare_ggn_vjp_compile_boundary(
            execution,
            settings,
            lambda output_cotangent: _run_ggnvp_vjp_by_path(
                execution,
                _ggn_tensor_function(execution),
                output_cotangent,
            ),
        )

    return None


def _require_compiled_execution(
    execution: StandardExecution,
    settings: Mapping[str, Any],
) -> None:
    _require_compile_boundary(execution.operator, settings)

    if _compile_bool(settings, "compile.compiled_autograd"):
        _require_compiled_autograd_operator(execution.operator)


def _prepare_inner_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    builder: CandidateOperation,
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    compiled_inner = _compiled_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_inner=compiled_inner,
    )


def _prepare_model_forward_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)

    if execution.module is None or execution.module_call is None:
        message = "compile.boundary=model_forward requires module_call"
        raise MaterializationError(message)

    module = execution.module
    module_call = execution.module_call

    def model_forward(batch: Batch) -> object:
        return _invoke_stateful_module(
            module,
            module_call,
            batch,
        )

    compiled_model_forward = _compiled_model_forward(settings, model_forward)

    if settings.get("compile.cache_state") == "warm_cache":
        _call_compiled_model_forward(
            execution,
            compiled_model_forward,
            execution.params,
        )

    return dataclasses.replace(
        execution,
        compiled_model_forward=compiled_model_forward,
    )


def _prepare_loss_closure_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    scalar_function = _hvp_scalar_function(execution)
    compiled_scalar_function = _compiled_scalar_function(
        settings,
        scalar_function,
        execution.params,
    )

    return dataclasses.replace(
        execution,
        compiled_scalar_function=compiled_scalar_function,
    )


def _prepare_score_matrix_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[], torch.Tensor],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    compiled_score_matrix = _compiled_tensor_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_score_matrix=compiled_score_matrix,
    )


def _prepare_ggn_loss_product_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[TensorTree, TensorTree], TensorTree],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    warm_inputs = (
        _ggn_loss_product_warm_inputs(execution)
        if settings.get("compile.cache_state") == "warm_cache"
        else None
    )
    compiled_loss_product = _compiled_ggn_loss_product_operation(
        settings,
        builder,
        None if warm_inputs is None else warm_inputs[0],
        None if warm_inputs is None else warm_inputs[1],
    )

    return dataclasses.replace(
        execution,
        compiled_ggn_loss_product=compiled_loss_product,
    )


def _prepare_ggn_jvp_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[], tuple[TensorTree, TensorTree]],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    compiled_ggn_jvp = _compiled_ggn_jvp_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_ggn_jvp=compiled_ggn_jvp,
    )


def _prepare_ggn_vjp_compile_boundary(
    execution: StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[TensorTree], TensorTree],
) -> StandardExecution:
    _require_compiled_execution(execution, settings)
    warm_output_cotangent = (
        _ggn_vjp_warm_input(execution)
        if settings.get("compile.cache_state") == "warm_cache"
        else None
    )
    compiled_vjp = _compiled_ggn_vjp_operation(
        settings,
        builder,
        warm_output_cotangent,
    )

    return dataclasses.replace(
        execution,
        compiled_ggn_vjp=compiled_vjp,
    )


def _prepare_gradient_execution(execution: StandardExecution) -> StandardExecution:
    schedule = execution.candidate.settings.get("gradient.graph_schedule")

    if schedule is None or schedule == "rebuild_per_call":
        return execution

    if schedule != "build_once":
        message = f"gradient.graph_schedule is unsupported: {schedule}"
        raise MaterializationError(message)

    return dataclasses.replace(
        execution,
        prepared_gradient=_gradient_operation_by_path(execution),
    )


def _prepare_jvp_execution(execution: StandardExecution) -> StandardExecution:
    reuse = execution.candidate.settings.get("jvp.linearize_reuse")

    if reuse is None or reuse == "none":
        return execution

    if reuse != "reuse_at_same_primal":
        message = f"jvp.linearize_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != JVP_LINEARIZE_PATH:
        message = "reuse_at_same_primal requires torch_func_linearize"
        raise MaterializationError(message)

    _, jvp_function = torch.func.linearize(
        _jvp_tensor_function(execution),
        execution.params,
    )

    return dataclasses.replace(execution, linearized_jvp=jvp_function)


def _prepare_vjp_execution(execution: StandardExecution) -> StandardExecution:
    reuse = execution.candidate.settings.get("vjp.closure_reuse")

    if reuse is None or reuse == "none":
        return execution

    if reuse != "reuse_vjp_closure_at_same_primal":
        message = f"vjp.closure_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != VJP_PATH:
        message = "reuse_vjp_closure_at_same_primal requires torch_func_vjp"
        raise MaterializationError(message)

    pullback = _vjp_pullback(
        _vjp_tensor_function(execution),
        execution.params,
    )

    def closure(cotangent: TensorTree) -> TensorTree:
        (result,) = pullback(cotangent)

        return result

    return dataclasses.replace(execution, vjp_closure=closure)


def _prepare_hvp_execution(execution: StandardExecution) -> StandardExecution:
    reuse = execution.candidate.settings.get("hvp.gradient_reuse")

    if reuse is None or reuse == "recompute_gradient":
        return execution

    if reuse != "reuse_gradient_closure":
        message = f"hvp.gradient_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != HVP_LINEARIZE_GRAD_PATH:
        message = "reuse_gradient_closure requires linearize_grad"
        raise MaterializationError(message)

    gradient_function = torch.func.grad(_hvp_scalar_function(execution))
    _, hvp_function = torch.func.linearize(gradient_function, execution.params)

    return dataclasses.replace(execution, linearized_hvp=hvp_function)


def _activation_operation(
    execution: StandardExecution,
    operation: CandidateOperation,
) -> CandidateOperation:
    settings = execution.candidate.settings

    if not _has_activation_settings(settings):
        return operation

    if settings.get("activation.recompute") == "manual_recompute":
        if execution.manual_recompute is None:
            message = "manual_recompute requires a declared recompute callback"
            raise MaterializationError(message)

        recomputed_operation = execution.manual_recompute(
            execution.candidate,
            operation,
            _activation_tensor_args(execution),
        )

        return _with_activation_offload(
            execution.candidate,
            recomputed_operation,
            _activation_offload(execution.candidate),
            execution.activation_pack_hooks,
            execution.activation_unpack_hooks,
        )

    def function(*_: torch.Tensor) -> TensorTree:
        return operation()

    try:
        return checkpoint_operation(
            execution.candidate,
            function,
            _activation_tensor_args(execution),
            policy_key="activation.recompute",
            activation_pack_hooks=execution.activation_pack_hooks,
            activation_unpack_hooks=execution.activation_unpack_hooks,
            checkpoint_contexts=execution.checkpoint_contexts,
        )
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error


def _activation_tensor_args(execution: StandardExecution) -> tuple[torch.Tensor, ...]:
    return (
        *_tensor_args(execution.params),
        *_tensor_args(execution.buffers),
        *_tensor_args(execution.batch),
        *_tensor_args(execution.vector),
    )


def _tensor_args(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)

    if isinstance(value, Mapping):
        return tuple(
            tensor for child in value.values() for tensor in _tensor_args(child)
        )

    if isinstance(value, tuple):
        return tuple(tensor for child in value for tensor in _tensor_args(child))

    return ()


def _require_finite_execution_inputs(execution: StandardExecution) -> None:
    for name, value in (
        ("parameters", execution.params),
        ("buffers", execution.buffers),
        ("batch", execution.batch),
        ("vector", execution.vector),
    ):
        _require_finite_nested_tensors(value, name)


def _require_finite_nested_tensors(value: Any, name: str) -> None:
    for tensor in _tensor_args(value):
        _require_finite_tensor(tensor, name)


def _has_activation_settings(settings: Mapping[str, Any]) -> bool:
    return any(key.startswith(("activation.", "checkpoint.")) for key in settings)


def _execution_with_inside_input_residency(
    execution: StandardExecution,
) -> StandardExecution:
    if _move_input_residency_outside_measured_call(execution.candidate.settings):
        return execution

    return dataclasses.replace(
        execution,
        batch=_runtime_batch_input_residency(
            execution.batch,
            execution.candidate.settings,
        ),
    )


def _move_input_residency_outside_measured_call(
    settings: Mapping[str, Any],
) -> bool:
    value = settings.get("input.host_to_device")

    if value is None or value == "outside_measured_call":
        return True

    if value == "inside_measured_call":
        return False

    message = f"input.host_to_device is unsupported: {value}"
    raise MaterializationError(message)


def _compile_operation(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    operation: CandidateOperation,
) -> CandidateOperation:
    enabled = settings.get("compile.enabled")

    if enabled is None or enabled == "false":
        return operation

    if enabled != "true":
        message = f"compile.enabled is unsupported: {enabled}"
        raise MaterializationError(message)

    _require_compile_boundary(operator, settings)

    compiled_autograd = _compile_bool(settings, "compile.compiled_autograd")
    if compiled_autograd:
        _require_compiled_autograd_operator(operator)

    if _compile_boundary_runs_inside_operator(operator.kind, settings):
        return operation

    return _compiled_operation(settings, operation)


def _compile_boundary_runs_inside_operator(
    operator_kind: str,
    settings: Mapping[str, Any],
) -> bool:
    if settings.get("compile.boundary") == "model_forward":
        return True

    return (operator_kind, settings.get("compile.boundary")) in {
        ("gradient", "loss_closure"),
        ("gradient", "gradient_closure"),
        ("jvp", "jvp_closure"),
        ("vjp", "vjp_closure"),
        ("hvp", "loss_closure"),
        ("hvp", "hvp_single_vector"),
        ("hvp", "hvp_batched_vectors"),
        ("ggnvp", "ggn_full_product"),
        ("ggnvp", "ggn_jvp"),
        ("ggnvp", "ggn_loss_hessian_product"),
        ("ggnvp", "ggn_vjp"),
        ("fisher_vp", "fisher_score_grad"),
        ("sampled_fisher_vp", "sampled_fisher_score_grad"),
        ("empirical_fisher_vp", "empirical_fisher_example_grad"),
        ("per_example_gradient", "per_example_gradient"),
        ("metric", "metric_multiply"),
        ("sqrt_metric", "metric_sqrt_multiply"),
        ("inverse_sqrt_metric", "metric_sqrt_multiply"),
        ("metric_inner", "metric_inner_reduce"),
        ("inverse_metric", "inverse_metric_solve"),
        ("inverse_metric_inner", "inverse_metric_inner_reduce"),
        ("composition", "composition_child"),
    }


def _compiled_operation(
    settings: Mapping[str, Any],
    operation: CandidateOperation,
) -> CandidateOperation:
    compiled_operation = _compiled_callable(
        settings,
        operation,
        use_backend_settings=True,
    )
    _warm_compiled_cache(settings, compiled_operation)

    return compiled_operation


def _validate_compile_cache_state(settings: Mapping[str, Any]) -> None:
    if settings.get("compile.cache_state") in {"cold_compile", "warm_cache"}:
        return

    message = "compile.cache_state must be cold_compile or warm_cache"
    raise MaterializationError(message)


def _compiled_tensor_operation(
    settings: Mapping[str, Any],
    operation: Callable[[], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    compiled_operation = _compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _warm_compiled_cache(settings, compiled_operation)

    return compiled_operation


def _compiled_model_forward(
    settings: Mapping[str, Any],
    operation: Callable[[Batch], object],
) -> Callable[[Batch], object]:
    compiled_function = _compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _validate_compile_cache_state(settings)

    return compiled_function


def _compiled_scalar_function(
    settings: Mapping[str, Any],
    operation: Callable[[ParameterTree], torch.Tensor],
    warm_params: ParameterTree,
) -> Callable[[ParameterTree], torch.Tensor]:
    compiled_function = _compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _warm_compiled_cache(settings, compiled_function, warm_params)

    return compiled_function


def _compiled_ggn_loss_product_operation(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree, TensorTree], TensorTree],
    warm_output: TensorTree | None,
    warm_output_jvp: TensorTree | None,
) -> Callable[[TensorTree, TensorTree], TensorTree]:
    compiled_operation = _compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _warm_compiled_cache(
        settings,
        compiled_operation,
        warm_output,
        warm_output_jvp,
        error_message="GGN loss-product warm cache requires warm inputs",
    )

    return compiled_operation


def _compiled_ggn_jvp_operation(
    settings: Mapping[str, Any],
    operation: Callable[[], tuple[TensorTree, TensorTree]],
) -> Callable[[], tuple[TensorTree, TensorTree]]:
    compiled_operation = _compiled_callable(
        settings,
        operation,
        use_backend_settings=True,
    )
    _warm_compiled_cache(settings, compiled_operation)

    return compiled_operation


def _compiled_ggn_vjp_operation(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree], TensorTree],
    warm_output_cotangent: TensorTree | None,
) -> Callable[[TensorTree], TensorTree]:
    compiled_operation = _compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _warm_compiled_cache(
        settings,
        compiled_operation,
        warm_output_cotangent,
        error_message="GGN VJP warm cache requires warm output cotangent",
    )

    return compiled_operation


def _compiled_callable(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
    *,
    use_backend_settings: bool,
) -> Callable[..., Any]:
    compiled_autograd = _compile_bool(settings, "compile.compiled_autograd")
    compiled = _compile_with_runtime_settings(
        settings,
        operation,
        compiled_autograd=compiled_autograd,
    )

    def compiled_function(*args: Any) -> Any:
        if compiled_autograd:
            with _compiled_autograd_patch():
                return _call_compiled_callable(
                    settings,
                    compiled,
                    args,
                    use_backend_settings=use_backend_settings,
                )

        return _call_compiled_callable(
            settings,
            compiled,
            args,
            use_backend_settings=use_backend_settings,
        )

    return compiled_function


def _compile_with_runtime_settings(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
    *,
    compiled_autograd: bool,
) -> Callable[..., Any]:
    if compiled_autograd:
        with _compiled_autograd_patch():
            return _torch_compile_with_runtime_settings(settings, operation)

    return _torch_compile_with_runtime_settings(settings, operation)


def _torch_compile_with_runtime_settings(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
) -> Callable[..., Any]:
    return torch.compile(
        operation,
        backend=_compile_backend(settings),
        mode=_compile_mode(settings),
        fullgraph=_compile_bool(settings, "compile.fullgraph"),
        dynamic=_compile_optional_bool(settings, "compile.dynamic"),
        options=_compile_options(settings),
    )


def _call_compiled_callable(
    settings: Mapping[str, Any],
    compiled: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    use_backend_settings: bool,
) -> Any:
    if use_backend_settings:
        return _call_compiled_operation(settings, compiled, *args)

    return _call_with_deferred_finite_checks(compiled, *args)


def _warm_compiled_cache(
    settings: Mapping[str, Any],
    compiled: Callable[..., Any],
    *warm_args: Any,
    error_message: str | None = None,
) -> None:
    cache_state = settings.get("compile.cache_state")
    _validate_compile_cache_state(settings)

    if cache_state != "warm_cache":
        return

    if error_message is not None and any(arg is None for arg in warm_args):
        raise MaterializationError(error_message)

    compiled(*warm_args)


def _require_compile_boundary(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    boundary = settings.get("compile.boundary")

    if not isinstance(boundary, str):
        message = "compile.boundary is required"
        raise MaterializationError(message)

    if _compile_boundary_supported(operator.kind, boundary, settings):
        return

    message = f"compile.boundary={boundary} is not lowered for {operator.kind}"
    raise MaterializationError(message)


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

    if boundary == "ggn_jvp":
        return settings.get("ggn.jvp_path") in {
            "torch_func_jvp",
            "forward_ad_dual",
            "torch_func_linearize",
        }

    if boundary == "ggn_loss_hessian_product":
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
    row = SCORE_MATRIX_COMPILE_ROWS[operator_kind]
    expected_boundary, path_key = row

    if boundary != expected_boundary:
        return False

    if (
        operator_kind == "per_example_gradient"
        and settings.get("per_example_gradient.accumulation") != "stacked_leading_axis"
    ):
        return False

    return settings.get(path_key) in SCORE_MATRIX_COMPILE_PATH_VALUES


def _compiled_autograd_patch() -> Any:
    config = importlib.import_module("torch._dynamo.config")

    return config.patch({"compiled_autograd": True})


def _require_compiled_autograd_operator(operator: OperatorSpec) -> None:
    if operator.kind in {
        "gradient",
        "vjp",
        "hvp",
        "ggnvp",
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return

    message = (
        "compile.compiled_autograd=true requires a backward or higher-order "
        f"operator, got {operator.kind}"
    )
    raise MaterializationError(message)


def _compile_backend(settings: Mapping[str, Any]) -> str:
    value = settings.get("compile.backend")

    if not isinstance(value, str):
        message = "compile.backend is required"
        raise MaterializationError(message)

    if value == "inductor":
        return value

    if value == "registered_backend":
        message = "compile.backend requires a concrete PyTorch compiler backend id"
        raise MaterializationError(message)

    if _is_registered_compile_backend(value):
        return value

    message = f"compile.backend is not registered with PyTorch: {value}"
    raise MaterializationError(message)


def _is_registered_compile_backend(value: str) -> bool:
    try:
        backends = torch.compiler.list_backends()
    except AttributeError as error:
        message = "torch.compiler.list_backends is required"
        raise MaterializationError(message) from error

    return value in set(backends)


def _compile_mode(settings: Mapping[str, Any]) -> str | None:
    value = settings.get("compile.mode")

    if value is None:
        return None

    if value in {"default", "max-autotune"}:
        return value

    message = f"compile.mode is unsupported: {value}"
    raise MaterializationError(message)


def _compile_bool(settings: Mapping[str, Any], key: str) -> bool:
    value = settings.get(key)

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be true or false"
    raise MaterializationError(message)


def _compile_optional_bool(settings: Mapping[str, Any], key: str) -> bool | None:
    value = settings.get(key)

    if value is None:
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be None, true, or false"
    raise MaterializationError(message)


def _compile_options(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    options = {}

    if _compile_bool(settings, "compile.options.epilogue_fusion"):
        options["epilogue_fusion"] = True

    if _compile_bool(settings, "compile.options.shape_padding"):
        options["shape_padding"] = True

    if _compile_bool(settings, "compile.cuda_graphs"):
        options["triton.cudagraphs"] = True

    if not options:
        return None

    return options


def _apply_numeric_error_bound(
    measurements: dict[str, float],
    thresholds: Mapping[str, float],
    settings: Mapping[str, Any],
    bound_fields: Mapping[str, Any],
    reference: TensorTree,
) -> None:
    bound_measurements = numeric_error_bound_measurements(
        settings,
        bound_fields,
        reference,
    )
    validate_numeric_error_bound(measurements, thresholds, bound_measurements)
    measurements.update(bound_measurements)


def standard_reference_check(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    thresholds: Mapping[str, float],
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: ActivationPackHooks | None = None,
    activation_unpack_hooks: ActivationUnpackHooks | None = None,
    checkpoint_contexts: CheckpointContextFns | None = None,
) -> ReferenceCheck:
    """Return a reference check backed by package-owned anchors.

    Raises:
        MaterializationError: If thresholds are empty.
    """
    if not thresholds:
        message = "reference thresholds are required"
        raise MaterializationError(message)

    scalar_map = {} if scalar_objectives is None else dict(scalar_objectives)
    function_map = {} if function_objectives is None else dict(function_objectives)
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    candidate_factory = standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_map,
        function_objectives=function_map,
        module=module,
        module_call=module_call,
        teacher_objective=teacher_objective,
        batch_layout=batch_layout,
        lm_head_chunker=lm_head_chunker,
        fusion_rewriter=fusion_rewriter,
        mmap_residency=mmap_residency,
        manual_recompute=manual_recompute,
        activation_pack_hooks=activation_pack_hooks,
        activation_unpack_hooks=activation_unpack_hooks,
        checkpoint_contexts=checkpoint_contexts,
    )

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        try:
            anchor_candidate, candidate_output, anchor_output = (
                _standard_reference_outputs(
                    operator,
                    candidate,
                    batch,
                    vector,
                    thresholds,
                    candidate_factory,
                )
            )
        except ReferenceFailedError:
            raise
        except RuntimeError as error:
            raise ReferenceFailedError(str(error)) from error

        try:
            measurements = _standard_reference_measurements(
                operator,
                candidate,
                anchor_candidate,
                batch,
                vector,
                candidate_output,
                anchor_output,
                candidate_factory,
                params=params,
                buffers=buffers,
                scalar_objectives=scalar_map,
                function_objectives=function_map,
                parameter_surface=parameter_surface,
            )
        except ReferenceFailedError:
            raise
        except RuntimeError as error:
            raise ReferenceFailedError(str(error)) from error

        effective_thresholds = _reference_thresholds_for_operator(operator, thresholds)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)
        _apply_numeric_error_bound(
            measurements,
            effective_thresholds,
            candidate.settings,
            bound_fields,
            anchor_output,
        )

        return ReferenceResult(
            "standard_anchor",
            effective_thresholds,
            measurements,
        )

    return check


def _reference_thresholds_for_operator(
    operator: OperatorSpec,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    effective_thresholds = dict(thresholds)

    if operator.kind in {"inverse_metric", "inverse_metric_inner"}:
        tolerance = _inverse_metric_tolerance(operator)

        if tolerance is not None:
            effective_thresholds["inverse_residual"] = tolerance

    return effective_thresholds


def _standard_reference_outputs(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    thresholds: Mapping[str, float],
    candidate_factory: OperationFactory,
) -> tuple[Candidate, TensorTree, TensorTree]:
    anchor_candidate = _anchor_candidate(operator, candidate)
    _require_batch_inputs(operator, candidate, batch, phase="reference")
    _require_batch_inputs(operator, anchor_candidate, batch, phase="reference")
    _require_vhp_reference_policy(candidate, batch, thresholds)
    candidate_output = candidate_factory(candidate, batch, vector)()

    if (
        operator.kind
        in {"metric", "inverse_metric", "sqrt_metric", "inverse_sqrt_metric"}
        and _metric_representation_kind(operator) == "matrix_free"
    ):
        anchor_output = candidate_output
    elif operator.kind in {"metric", "inverse_metric"}:
        anchor_output = _metric_reference_output(operator, batch, vector)
    else:
        anchor_output = candidate_factory(anchor_candidate, batch, vector)()

    return anchor_candidate, candidate_output, anchor_output


def _require_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    *,
    phase: str,
) -> None:
    required = _required_batch_inputs(operator, candidate, phase)
    missing = tuple(key for key in required if key not in batch)

    if missing:
        message = (
            f"batch inputs missing for {operator.family}/{candidate.candidate_id}/"
            f"{phase}: {missing}"
        )
        raise MaterializationError(message)


def _required_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    phase: str,
) -> tuple[str, ...]:
    declared = operator.batch_inputs.get(phase)

    if declared is None:
        message = f"operator batch_inputs must declare {phase}"
        raise MaterializationError(message)

    declared = _ggn_declared_batch_inputs(operator, candidate, declared, phase)
    path_inputs = _candidate_batch_inputs(operator, candidate)
    teacher_inputs = _teacher_output_batch_inputs(candidate)

    return tuple(dict.fromkeys((*declared, *path_inputs, *teacher_inputs)))


def _teacher_output_batch_inputs(candidate: Candidate) -> tuple[str, ...]:
    if "teacher_outputs" not in candidate.settings:
        return ()

    return ("teacher_outputs",)


def _ggn_declared_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    declared: tuple[str, ...],
    phase: str,
) -> tuple[str, ...]:
    if operator.kind != "ggnvp":
        return declared

    if phase != "operation":
        return declared

    if candidate.settings.get("ggn.loss_hessian_path") != "closed_form_softmax_ce_kl":
        return declared

    return tuple(key for key in declared if key != "loss_hessian")


def _candidate_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
) -> tuple[str, ...]:
    path = _runtime_path(operator, candidate)

    if operator.kind == "fisher_vp":
        return _fisher_batch_inputs(operator, path)

    if operator.kind == "sampled_fisher_vp":
        return _sampled_fisher_batch_inputs(operator, path)

    if operator.kind == "empirical_fisher_vp":
        return _empirical_fisher_batch_inputs(operator, path)

    return ()


def _fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    if path not in {
        FISHER_DENSE_PATH,
        FISHER_SCORE_GRADIENT_LOOP_PATH,
    }:
        return ()

    inputs = ()

    if path == FISHER_DENSE_PATH:
        inputs = ("score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator))


def _sampled_fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    if path not in {
        SAMPLED_FISHER_DENSE_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    }:
        return ()

    inputs = ()

    if path == SAMPLED_FISHER_DENSE_PATH:
        inputs = ("sampled_score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator))


def _fisher_denominator_batch_inputs(operator: OperatorSpec) -> tuple[str, ...]:
    denominator = _operator_semantic(operator, "denominator")

    if denominator == "batch_normalization":
        return ("normalization",)

    if denominator == "num_examples":
        return ("num_examples",)

    return ()


def _empirical_fisher_batch_inputs(
    operator: OperatorSpec,
    path: str,
) -> tuple[str, ...]:
    if path not in {
        EMPIRICAL_FISHER_DENSE_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
    }:
        return ()

    inputs = ()

    if path == EMPIRICAL_FISHER_DENSE_PATH:
        inputs = ("per_example_gradients",)

    if _operator_semantic(operator, "denominator") == "batch_normalization":
        return (*inputs, "normalization")

    return inputs


def _standard_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    anchor_candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    anchor_output: TensorTree,
    candidate_factory: OperationFactory,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    function_objectives: Mapping[str, FunctionObjective],
    parameter_surface: ParameterSurface | None,
) -> dict[str, Any]:
    measurements = _layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        anchor_output,
    )
    _augment_ggn_dense_cross_check(
        operator,
        candidate,
        batch,
        vector,
        candidate_output,
        candidate_factory,
        measurements,
    )
    measurements.update(
        _semantic_measurements(operator, batch, vector, candidate_output)
    )
    measurements.update(
        _matrix_free_inverse_reference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
        )
    )
    measurements.update(
        _ggn_inner_product_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            anchor_candidate,
            candidate_factory,
            parameter_surface,
        )
    )
    measurements.update(
        _first_order_reference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            params=params,
            buffers=buffers,
            scalar_objectives=scalar_objectives,
            function_objectives=function_objectives,
        )
    )
    measurements.update(
        _hvp_finite_difference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            params=params,
            buffers=buffers,
            scalar_objectives=scalar_objectives,
            anchor_candidate=anchor_candidate,
            candidate_factory=candidate_factory,
            parameter_surface=parameter_surface,
        )
    )

    return measurements


def _augment_ggn_dense_cross_check(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    candidate_factory: OperationFactory,
    measurements: dict[str, Any],
) -> None:
    if operator.kind != "ggnvp":
        return

    dense_candidate = dataclasses.replace(
        candidate,
        settings=_anchor_settings(operator, candidate, GGN_DENSE_PATH),
    )
    dense_output = candidate_factory(dense_candidate, batch, vector)()
    errors = _layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        dense_output,
    )
    measurements["max_abs_diff"] = max(
        float(measurements["max_abs_diff"]),
        float(errors["max_abs_diff"]),
    )
    measurements["max_rel_diff"] = max(
        float(measurements["max_rel_diff"]),
        float(errors["max_rel_diff"]),
    )
    measurements["dense_anchor_errors"] = dict(errors)


def _layout_aware_tree_error_measurements(
    candidate: Candidate,
    candidate_output: TensorTree,
    anchor_output: TensorTree,
) -> dict[str, float]:
    if candidate.settings.get("layout.output") != "flat_contiguous":
        return tree_error_measurements(candidate_output, anchor_output)

    return tree_error_measurements(
        _flatten_vector(candidate_output),
        _flatten_vector(anchor_output),
    )


def _layout_aware_tree_dot(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    if settings.get("layout.output") != "flat_contiguous":
        return _tree_dot_runtime(settings, left, right)

    return _dot_runtime(settings, _flatten_vector(left), _flatten_vector(right))


def _runtime_callable_identity(
    value: Callable[..., Any] | None,
    name: str,
) -> Any:
    if value is None:
        return None

    return _callable_identity(value, name)


def _runtime_callable_map_identity(
    values: Mapping[str, Callable[..., Any]] | None,
    name: str,
) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()

    return tuple(
        {
            "id": key,
            "identity": _callable_identity(callback, f"{name}.{key}"),
        }
        for key, callback in sorted(values.items())
    )


def _callable_identity(value: Callable[..., Any], name: str) -> Any:
    explicit_identity = _explicit_callable_identity(value)

    if explicit_identity is not None:
        return explicit_identity

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)

    if not isinstance(module, str) or not isinstance(qualname, str):
        message = f"{name} must provide identity() or signature()"
        raise MaterializationError(message)

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        message = f"{name} must provide identity() or signature()"
        raise MaterializationError(message) from error

    return {
        "kind": "python_callable",
        "module": module,
        "qualname": qualname,
        "source_hash": stable_hash({"source": source}),
        "defaults": _json_identity(getattr(value, "__defaults__", None), name),
        "kwdefaults": _json_identity(getattr(value, "__kwdefaults__", None), name),
        "closure": _callable_closure_identity(value, name),
    }


def _explicit_callable_identity(value: Callable[..., Any]) -> Any | None:
    identity = getattr(value, "identity", None)

    if callable(identity):
        return {
            "kind": "explicit_identity",
            "value": _json_identity(identity(), "callable.identity"),
        }

    signature = getattr(value, "signature", None)

    if callable(signature):
        return {
            "kind": "explicit_signature",
            "value": _json_identity(signature(), "callable.signature"),
        }

    return None


def _json_identity(value: Any, name: str) -> Any:
    try:
        return to_json_value(value)
    except TypeError as error:
        message = f"{name} must be JSON-compatible"
        raise MaterializationError(message) from error


def _callable_closure_identity(value: Callable[..., Any], name: str) -> tuple[Any, ...]:
    closure = getattr(value, "__closure__", None)

    if closure is None:
        return ()

    if closure:
        message = f"{name} closes over runtime state; provide identity() or signature()"
        raise MaterializationError(message)

    return ()


def _keyword_map(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def standard_runtime_config(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: ActivationPackHooks | None = None,
    activation_unpack_hooks: ActivationUnpackHooks | None = None,
    checkpoint_contexts: CheckpointContextFns | None = None,
) -> RuntimeConfig:
    """Return runtime config for package-owned standard operators."""
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    mmap_residency_callback = mmap_residency
    runtime_bindings = _keyword_map(
        params=params,
        buffers=buffers,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=module,
        module_call=module_call,
        teacher_objective=teacher_objective,
        batch_layout=batch_layout,
        lm_head_chunker=lm_head_chunker,
        fusion_rewriter=fusion_rewriter,
        mmap_residency=mmap_residency_callback,
        manual_recompute=manual_recompute,
        activation_pack_hooks=activation_pack_hooks,
        activation_unpack_hooks=activation_unpack_hooks,
        checkpoint_contexts=checkpoint_contexts,
    )
    operation_factory = standard_operation_factory(operator, **runtime_bindings)
    reference_check = standard_reference_check(
        operator,
        thresholds=thresholds,
        numeric_bound_fields=bound_fields,
        **runtime_bindings,
    )
    runtime_signature = {
        "runtime": "standard",
        "operator": operator.signature(),
        "params": tree_signature(params),
        "buffers": tree_signature(buffers),
        "parameter_surface": (
            None if parameter_surface is None else parameter_surface.signature()
        ),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": bound_fields,
        "objective": dict(objective_signature),
        "module": module is not None,
        "module_call": None if module_call is None else module_call.signature(),
        "teacher_objective": _runtime_callable_identity(
            teacher_objective,
            "teacher_objective",
        ),
        "batch_layout": _runtime_callable_identity(batch_layout, "batch_layout"),
        "lm_head_chunker": _runtime_callable_identity(
            lm_head_chunker,
            "lm_head_chunker",
        ),
        "fusion_rewriter": _runtime_callable_identity(
            fusion_rewriter,
            "fusion_rewriter",
        ),
        "mmap_residency": _runtime_callable_identity(
            mmap_residency,
            "mmap_residency",
        ),
        "manual_recompute": _runtime_callable_identity(
            manual_recompute,
            "manual_recompute",
        ),
        "activation_pack_hooks": _runtime_callable_map_identity(
            activation_pack_hooks,
            "activation_pack_hooks",
        ),
        "activation_unpack_hooks": _runtime_callable_map_identity(
            activation_unpack_hooks,
            "activation_unpack_hooks",
        ),
        "checkpoint_contexts": _runtime_callable_map_identity(
            checkpoint_contexts,
            "checkpoint_contexts",
        ),
    }
    operation_factory = CallableOperationFactory(
        "vptune.standard_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.standard_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.standard_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.standard_reference_check"},
        reference_check,
    )
    materializer = _standard_materializer(
        operation_factory,
        operator,
        mmap_residency_callback,
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        reference_check_name="standard_anchor",
        signature=runtime_signature,
    )


def standard_problem(
    *,
    model: torch.nn.Module,
    parameter_surface: ParameterSurface,
    parameter_values: ParameterTree,
    buffers: BufferTree,
    data: DataProvider,
    operator: OperatorSpec,
    vectors: VectorProvider,
    target: Target,
    candidates: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module_call: ModuleCallSpec | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
) -> Problem:
    """Return a standard PyTorch tuning problem from explicit settings.

    Raises:
        MaterializationError: If candidate settings are empty.
    """
    if operator.kind == "composition":
        message = "standard_problem requires TuningRun for composition operators"
        raise MaterializationError(message)

    if not candidates:
        message = "problem candidate_settings are required"
        raise MaterializationError(message)

    axis_registry = standard_axis_registry()
    candidate_rows = tuple(
        Candidate(
            family=operator.family,
            candidate_id=candidate_id,
            settings=dict(settings),
            changed_axes=_standard_changed_axes(settings, axis_registry),
            generator_id="vptune.problem",
            generator_version=PACKAGE_VERSION,
        )
        for candidate_id, settings in candidates.items()
    )
    runtime = standard_runtime_config(
        operator,
        params=parameter_values,
        buffers=buffers,
        candidates=candidate_rows,
        thresholds=thresholds,
        objective_signature=objective_signature,
        axis_registry=axis_registry,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=model,
        module_call=module_call,
        batch_layout=batch_layout,
    )

    return Problem(
        model=model,
        params=parameter_surface,
        data=data,
        operator=operator,
        vectors=vectors,
        target=target,
        runtime=runtime,
        anchor_policy={},
        replay_policy={},
        adapter_identity={
            "adapter_id": "vptune.core",
            "adapter_version": PACKAGE_VERSION,
        },
    )


def _standard_changed_axes(
    settings: Mapping[str, Any],
    axis_registry: Any,
) -> tuple[str, ...]:
    axes = set()

    for key in settings:
        owner = axis_registry.owners.get(key)

        if owner is not None:
            axes.add(owner)
            continue

        optional_owners = axis_registry.optional_owners.get(key)

        if optional_owners is not None:
            axes.update(optional_owners)
            continue

        axes.add(key)

    return tuple(sorted(axes))


def _run_standard_operation(execution: StandardExecution) -> TensorTree:
    _require_loss_scaling_settings(execution.operator, execution.candidate.settings)
    runner = STANDARD_RUNNERS.get(execution.operator.kind)

    if runner is None:
        message = (
            "standard runtime does not support operator kind: "
            f"{execution.operator.kind}"
        )
        raise MaterializationError(message)

    execution = _execution_with_recomputed_teacher_outputs(execution)

    if _uses_microbatch_accumulation(execution):
        return _run_microbatch_accumulate(execution)

    result = runner(execution)
    result = _loss_scaled_output_source(
        execution.operator,
        execution.candidate.settings,
        result,
    )

    return _loss_unscaled_output(
        execution.operator,
        execution.candidate.settings,
        result,
    )


def _loss_scaled_execution(execution: StandardExecution) -> StandardExecution:
    scale = _loss_scale(execution.candidate.settings)

    if scale is None:
        return execution

    if execution.operator.kind in {"gradient", "hvp"}:
        return dataclasses.replace(
            execution,
            scalar_objectives=_scaled_scalar_objectives(execution, scale),
        )

    if execution.operator.kind in {"jvp", "vjp"}:
        return dataclasses.replace(
            execution,
            function_objectives=_scaled_function_objectives(execution, scale),
        )

    if execution.operator.kind == "ggnvp":
        return dataclasses.replace(
            execution,
            batch=_scaled_loss_hessian_batch(execution.batch, scale),
        )

    return execution


def _scaled_scalar_objectives(
    execution: StandardExecution,
    scale: float,
) -> Mapping[str, ScalarObjective]:
    objective_id = execution.operator.objective_id
    objective = _scalar_objective(execution.operator, execution.scalar_objectives)
    objectives = dict(execution.scalar_objectives)

    def scaled(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> torch.Tensor:
        return objective(params, buffers, batch, context) * scale

    objectives[objective_id] = scaled

    return objectives


def _scaled_function_objectives(
    execution: StandardExecution,
    scale: float,
) -> Mapping[str, FunctionObjective]:
    objective_id = execution.operator.objective_id
    objective = _function_objective(execution.operator, execution.function_objectives)
    objectives = dict(execution.function_objectives)

    def scaled(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> TensorTree:
        output = objective(params, buffers, batch, context)
        output = _checked_function_output(
            execution.candidate.settings,
            output,
            "function objective output",
        )

        return tree_map(
            lambda tensor: tensor * scale,
            output,
        )

    objectives[objective_id] = scaled

    return objectives


def _scaled_loss_hessian_batch(batch: Batch, scale: float) -> Batch:
    if "loss_hessian" not in batch:
        return batch

    result = dict(batch)
    result["loss_hessian"] = _scaled_tensor(
        batch["loss_hessian"],
        scale,
        "loss_hessian",
    )

    return result


def _loss_scaled_output_source(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    scale = _loss_scale(settings)

    if scale is None:
        return result

    if operator.kind in {"metric", "inverse_metric", "composition"}:
        return _tree_scale_runtime(settings, result, scale)

    return result


def _loss_unscaled_output(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    scale = _loss_scale(settings)

    if scale is None:
        return result

    degree = _loss_unscale_degree(operator, settings)

    return _tree_scale_runtime(settings, result, 1.0 / (scale**degree))


def _require_loss_scaling_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    mode = settings.get("numeric.loss_scaling")
    has_scale = "numeric.loss_scale" in settings
    has_degree = "numeric.loss_unscale_degree" in settings

    if mode is None:
        if has_scale or has_degree:
            message = "numeric.loss_scaling is required for loss-scale fields"
            raise MaterializationError(message)

        return

    if mode == "none":
        if has_scale or has_degree:
            message = "numeric.loss_scaling=none forbids loss-scale fields"
            raise MaterializationError(message)

        return

    if mode != "static_scale_with_exact_unscale":
        message = f"numeric.loss_scaling is unsupported: {mode}"
        raise MaterializationError(message)

    _loss_scale(settings)
    _loss_unscale_degree(operator, settings)


def _loss_scale(settings: Mapping[str, Any]) -> float | None:
    mode = settings.get("numeric.loss_scaling")

    if mode is None or mode == "none":
        return None

    value = settings.get("numeric.loss_scale")

    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        message = "numeric.loss_scale must be a positive float"
        raise MaterializationError(message)

    return float(value)


def _loss_unscale_degree(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> int:
    value = settings.get("numeric.loss_unscale_degree")

    if isinstance(value, bool) or not isinstance(value, int):
        message = "numeric.loss_unscale_degree must be an integer"
        raise MaterializationError(message)

    expected = _expected_loss_unscale_degree(operator)

    if value != expected:
        message = (
            "numeric.loss_unscale_degree does not match operator: "
            f"{value} != {expected}"
        )
        raise MaterializationError(message)

    return value


def _expected_loss_unscale_degree(operator: OperatorSpec) -> int:
    if operator.kind in {
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return 2

    return 1


def _scaled_tensor(value: Any, scale: float, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        message = f"{name} must be a tensor"
        raise MaterializationError(message)

    return value * scale


def _composition_order(order: Sequence[str]) -> tuple[str, ...]:
    """Return validated composition order.

    Raises:
        MaterializationError: If the order is empty or contains duplicate names.
    """
    order_result = tuple(order)

    if not order_result:
        message = "composition order must be non-empty"
        raise MaterializationError(message)

    if len(set(order_result)) != len(order_result):
        message = "composition order contains duplicate component names"
        raise MaterializationError(message)

    return order_result


def _operator_composition_order(operator: OperatorSpec) -> tuple[str, ...]:
    if operator.kind != "composition":
        message = f"operator is not a composition: {operator.kind}"
        raise MaterializationError(message)

    children = operator.semantics.get("children")

    if not isinstance(children, Sequence) or isinstance(children, str):
        message = "composition operator must declare ordered children"
        raise MaterializationError(message)

    return _composition_order(children)


def _require_composition_components(
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
) -> None:
    if set(components) != set(order):
        message = "composition components must match composition order"
        raise MaterializationError(message)


def _run_gradient(execution: StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_gradient_by_path(execution)


def _run_gradient_by_path(execution: StandardExecution) -> TensorTree:
    if execution.prepared_gradient is not None:
        return execution.prepared_gradient()

    return _gradient_operation_by_path(execution)()


def _gradient_operation_by_path(execution: StandardExecution) -> CandidateOperation:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            GRADIENT_PATH,
            GRADIENT_TORCH_FUNC_PATH,
            GRADIENT_TORCH_FUNC_VALUE_PATH,
            GRADIENT_BACKWARD_MATERIALIZED_PATH,
        ),
    )
    scalar_function = _hvp_scalar_function(execution)

    if execution.path == GRADIENT_PATH:

        def operation() -> TensorTree:
            return gradient_anchor(scalar_function, execution.params)

        return operation

    if execution.path == GRADIENT_TORCH_FUNC_PATH:
        gradient_function = torch.func.grad(scalar_function)

        def operation() -> TensorTree:
            result = gradient_function(execution.params)
            _require_finite_tree(result, "gradient result")

            return result

        return operation

    if execution.path == GRADIENT_TORCH_FUNC_VALUE_PATH:
        gradient_function = torch.func.grad_and_value(scalar_function)

        def operation() -> TensorTree:
            result, value = gradient_function(execution.params)
            _require_finite_tree(result, "gradient result")
            _require_gradient_value_reuse(execution, value)

            return result

        return operation

    def operation() -> TensorTree:
        return _run_materialized_gradient(execution)

    return operation


def _uses_microbatch_accumulation(execution: StandardExecution) -> bool:
    return (
        execution.candidate.settings.get("schedule.gradient_accumulation")
        == "microbatch_accumulate"
    )


def _run_microbatch_accumulate(
    execution: StandardExecution,
) -> TensorTree:
    if execution.operator.aggregation != "sum":
        message = "microbatch_accumulate requires sum aggregation"
        raise MaterializationError(message)

    batch, batch_in_dims = _microbatch_in_dims(execution.batch)
    batch_size = _per_example_batch_size(
        batch,
        batch_in_dims,
        "microbatch accumulation",
    )
    microbatch_size = _data_microbatch_size(execution.candidate.settings)
    accumulated = None

    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        subbatch = _per_example_batch_slice(batch, batch_in_dims, start, stop)
        subexecution = dataclasses.replace(
            execution,
            batch=subbatch,
            candidate=dataclasses.replace(
                execution.candidate,
                settings=_single_step_microbatch_settings(execution.candidate.settings),
            ),
        )
        subresult = _run_standard_operation(subexecution)
        accumulated = (
            subresult
            if accumulated is None
            else _tree_add_runtime(
                execution.candidate.settings,
                accumulated,
                subresult,
            )
        )

    if accumulated is None:
        message = "microbatch accumulation requires a nonempty batch"
        raise MaterializationError(message)

    _require_finite_tree(accumulated, "microbatch result")

    return accumulated


def _single_step_microbatch_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(settings)
    result["schedule.gradient_accumulation"] = "single_step"
    result.pop("batch.data_microbatch_size", None)

    return result


def _microbatch_in_dims(batch: Batch) -> tuple[dict[str, Any], dict[str, int | None]]:
    return _per_example_batch_in_dims(batch, "microbatch accumulation")


def _require_gradient_value_reuse(
    execution: StandardExecution,
    value: torch.Tensor,
) -> None:
    reuse = execution.candidate.settings.get("gradient.value_reuse")

    if reuse is None or reuse == "gradient_only":
        return

    if reuse != "gradient_and_primal_value":
        message = f"gradient.value_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != GRADIENT_TORCH_FUNC_VALUE_PATH:
        message = "gradient_and_primal_value requires torch_func_grad_and_value"
        raise MaterializationError(message)

    _require_finite_tensor(value, "gradient primal value")


def _run_materialized_gradient(
    execution: StandardExecution,
) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)
    active_params = _grad_enabled_params(execution.params)
    value = scalar_function(active_params)
    value.backward()
    result = _parameter_grad_tree(active_params)
    _require_finite_tree(result, "gradient result")

    return result


def _run_jvp(execution: StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_jvp_by_path(execution)


def _run_by_vectorization_mode(
    execution: StandardExecution,
    *,
    single_vector: Callable[[StandardExecution], TensorTree],
    single_loop: Callable[[StandardExecution], TensorTree],
    manual_batch: Callable[[StandardExecution], TensorTree],
    vmap: Callable[[StandardExecution], TensorTree],
) -> TensorTree:
    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "single_loop":
        return single_loop(execution)

    if mode == "manual_batch":
        return manual_batch(execution)

    if mode == "vmap":
        return vmap(execution)

    return single_vector(execution)


def _run_single_vectorized_by_path(
    execution: StandardExecution,
    paths: tuple[str, ...],
    single_vector: Callable[[StandardExecution], TensorTree],
    single_loop_vector: Callable[[StandardExecution], TensorTree],
    vmap: Callable[[StandardExecution], TensorTree],
) -> TensorTree:
    _require_path(execution.operator.kind, execution.path, paths)

    def single_loop(loop_execution: StandardExecution) -> TensorTree:
        return _run_vector_single_loop(loop_execution, single_loop_vector)

    def manual_batch(batch_execution: StandardExecution) -> TensorTree:
        return _run_vector_manual_batches(batch_execution, single_loop)

    return _run_by_vectorization_mode(
        execution,
        single_vector=single_vector,
        single_loop=single_loop,
        manual_batch=manual_batch,
        vmap=vmap,
    )


def _run_jvp_by_path(execution: StandardExecution) -> TensorTree:
    return _run_single_vectorized_by_path(
        execution,
        (JVP_PATH, JVP_FORWARD_AD_PATH, JVP_LINEARIZE_PATH),
        _run_jvp_single_vector,
        _run_jvp_single_vector,
        _run_jvp_vector_vmap,
    )


def _run_jvp_single_vector(execution: StandardExecution) -> TensorTree:
    tensor_function = _jvp_tensor_function(execution)

    if execution.path == JVP_FORWARD_AD_PATH:
        return forward_ad_jvp_anchor(
            tensor_function,
            execution.params,
            execution.vector,
        )

    if execution.path == JVP_LINEARIZE_PATH:
        if execution.linearized_jvp is not None:
            return execution.linearized_jvp(execution.vector)

        _, jvp_function = torch.func.linearize(tensor_function, execution.params)

        return jvp_function(execution.vector)

    return jvp_anchor(tensor_function, execution.params, execution.vector)


def _run_jvp_vector_vmap(execution: StandardExecution) -> TensorTree:
    if execution.path not in JVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires a torch.func JVP path"
        raise MaterializationError(message)

    if execution.path == JVP_LINEARIZE_PATH:
        if execution.linearized_jvp is not None:
            jvp_function = execution.linearized_jvp
        else:
            _, jvp_function = torch.func.linearize(
                _jvp_tensor_function(execution),
                execution.params,
            )

        return _run_vector_vmap(execution, jvp_function)

    tensor_function = _jvp_tensor_function(execution)

    def jvp_function(vector: TensorTree) -> TensorTree:
        return jvp_anchor(
            tensor_function,
            execution.params,
            vector,
        )

    return _run_vector_vmap(execution, jvp_function)


def _jvp_tensor_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    function = _function_objective(execution.operator, execution.function_objectives)

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return _call_function_objective(execution, function, active_params)

    return tensor_function


def _run_vjp(execution: StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_vjp_by_path(execution)


def _run_vjp_by_path(execution: StandardExecution) -> TensorTree:
    return _run_single_vectorized_by_path(
        execution,
        (
            VJP_PATH,
            VJP_AUTOGRAD_OUTPUTS_PATH,
            VJP_BACKWARD_MATERIALIZED_PATH,
        ),
        _run_vjp_single_vector,
        _run_vjp_single_vector,
        _run_vjp_vector_vmap,
    )


def _run_vjp_single_vector(execution: StandardExecution) -> TensorTree:
    tensor_function = _vjp_tensor_function(execution)

    if execution.path == VJP_PATH:
        if execution.vjp_closure is not None:
            return execution.vjp_closure(execution.vector)

        pullback = _vjp_pullback(
            tensor_function,
            execution.params,
        )
        (result,) = pullback(execution.vector)

        return result

    return _run_autograd_vjp(execution, tensor_function)


def _run_vjp_vector_vmap(execution: StandardExecution) -> TensorTree:
    if execution.path not in VJP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires torch_func_vjp"
        raise MaterializationError(message)

    closure = execution.vjp_closure

    if closure is not None:

        def vjp_function(vector: TensorTree) -> TensorTree:
            return closure(vector)

    else:
        pullback = _vjp_pullback(
            _vjp_tensor_function(execution),
            execution.params,
        )

        def vjp_function(vector: TensorTree) -> TensorTree:
            (result,) = pullback(vector)

            return result

    return _run_vector_vmap(execution, vjp_function)


def _vjp_tensor_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    if _uses_stateful_module_call(execution):
        return _stateful_module_tensor_function(execution)

    function = _function_objective(execution.operator, execution.function_objectives)

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return _call_function_objective(execution, function, active_params)

    return tensor_function


def _vjp_pullback(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
) -> Callable[[TensorTree], tuple[TensorTree]]:
    vjp_result = torch.func.vjp(tensor_function, params, has_aux=False)

    return vjp_result[1]


def _run_autograd_vjp(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> TensorTree:
    if execution.path == VJP_AUTOGRAD_OUTPUTS_PATH:
        return _autograd_grad_outputs_vjp(
            tensor_function,
            execution.params,
            execution.vector,
        )

    return _backward_materialized_vjp(
        tensor_function,
        execution.params,
        execution.vector,
    )


def _backward_materialized_vjp(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    cotangent: TensorTree,
) -> TensorTree:
    active_params = _grad_enabled_params(params)
    output = tensor_function(active_params)
    output_leaves = tree_leaves(output)
    cotangent_leaves = tree_leaves(
        tree_map2(lambda out, cotangent: cotangent.reshape_as(out), output, cotangent)
    )

    torch.autograd.backward(output_leaves, grad_tensors=cotangent_leaves)
    result = _parameter_grad_tree(active_params)
    _require_finite_tree(result, "VJP result")

    return result


def _autograd_grad_outputs_vjp(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    cotangent: TensorTree,
) -> TensorTree:
    active_params = _grad_enabled_params(params)
    output = tensor_function(active_params)
    output_leaves = tree_leaves(output)
    cotangent_leaves = tree_leaves(
        tree_map2(lambda out, cotangent: cotangent.reshape_as(out), output, cotangent)
    )
    gradients = torch.autograd.grad(
        output_leaves,
        tuple(active_params.values()),
        grad_outputs=cotangent_leaves,
        allow_unused=True,
    )
    result = tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(param) if gradient is None else gradient.detach()
            for param, gradient in zip(active_params.values(), gradients, strict=True)
        ),
    )
    _require_finite_tree(result, "VJP result")

    return result


def _run_hvp(execution: StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_hvp_by_path(execution)


def _run_hvp_by_path(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            HVP_REFERENCE_PATH,
            HVP_FUNCTIONAL_PATH,
            HVP_JVP_GRAD_PATH,
            HVP_FORWARD_AD_PATH,
            HVP_LINEARIZE_GRAD_PATH,
            VHP_PATH,
        ),
    )

    return _run_by_vectorization_mode(
        execution,
        single_vector=_run_hvp_single_vector,
        single_loop=_run_hvp_vector_single_loop,
        manual_batch=_run_hvp_vector_manual_batch,
        vmap=_run_hvp_vector_vmap,
    )


def _run_hvp_single_vector(execution: StandardExecution) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)

    if execution.path == HVP_REFERENCE_PATH:
        if _hvp_row_batch_size(execution.candidate.settings) is None:
            result = hvp_reverse_over_reverse_anchor(
                scalar_function,
                execution.params,
                execution.vector,
            )
        else:
            result = _run_hvp_row_batched_reverse(execution, scalar_function)
    elif execution.path == HVP_FUNCTIONAL_PATH:
        result = hvp_anchor(scalar_function, execution.params, execution.vector)
    elif execution.path == VHP_PATH:
        result = _run_hvp_vhp_path(execution)
    elif execution.path == HVP_FORWARD_AD_PATH:
        result = _run_hvp_forward_ad_path(execution)
    elif execution.path == HVP_LINEARIZE_GRAD_PATH:
        if execution.linearized_hvp is not None:
            result = execution.linearized_hvp(execution.vector)
        else:
            gradient_function = torch.func.grad(scalar_function)
            _, hvp_function = torch.func.linearize(
                gradient_function,
                execution.params,
            )
            result = hvp_function(execution.vector)
    else:
        result = hvp_jvp_grad_anchor(
            scalar_function,
            execution.params,
            execution.vector,
        )

    return result


def _run_hvp_row_batched_reverse(
    execution: StandardExecution,
    scalar_function: Callable[[ParameterTree], torch.Tensor],
) -> TensorTree:
    batch_size = _hvp_row_batch_size(execution.candidate.settings)

    if batch_size is None:
        message = "batch.hvp_row_batch_size is required"
        raise MaterializationError(message)

    active_params = _grad_enabled_params(execution.params)
    parameter_leaves = tuple(active_params.values())
    value = scalar_function(active_params)
    gradient_leaves = torch.autograd.grad(
        value,
        parameter_leaves,
        create_graph=True,
        allow_unused=True,
    )
    gradient_flat = _flat_gradient_row(
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient
            for leaf, gradient in zip(parameter_leaves, gradient_leaves, strict=True)
        )
    )
    vector_tensor = _parameter_order_vector(execution)
    result = torch.zeros_like(vector_tensor)

    for start in range(0, gradient_flat.numel(), batch_size):
        stop = min(start + batch_size, gradient_flat.numel())

        for row_index in range(start, stop):
            component = gradient_flat[row_index]

            if not component.requires_grad:
                continue

            row_gradients = torch.autograd.grad(
                component,
                parameter_leaves,
                retain_graph=True,
                allow_unused=True,
            )
            row = _flat_gradient_row(
                tuple(
                    torch.zeros_like(leaf) if gradient is None else gradient
                    for leaf, gradient in zip(
                        parameter_leaves,
                        row_gradients,
                        strict=True,
                    )
                )
            )
            result[row_index] = _dot_runtime(
                execution.candidate.settings,
                row,
                vector_tensor,
            )

    _require_finite_tensor(result, "row-batched HVP result")

    return _wrap_flat_parameter_tree(active_params, result)


def _run_hvp_vector_single_loop(execution: StandardExecution) -> TensorTree:
    if _hvp_uses_reverse_reuse(execution.candidate.settings):
        return _run_hvp_reused_reverse_vectors(execution)

    return _run_vector_single_loop(execution, _run_hvp_single_vector)


def _run_hvp_vector_manual_batch(execution: StandardExecution) -> TensorTree:
    return _run_vector_manual_batches(execution, _run_hvp_vector_single_loop)


def _run_vector_manual_batches(
    execution: StandardExecution,
    runner: Callable[[StandardExecution], TensorTree],
) -> TensorTree:
    def vector_runner(vector: TensorTree) -> TensorTree:
        return runner(_execution_with_vector(execution, vector))

    return _run_tensor_tree_vector_manual_batches(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_tensor_tree_vector_manual_batches(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    vector_in_dims = _vector_tree_in_dims(
        vector_tree,
        settings,
    )
    vector_count = _vector_tree_batch_size(vector_tree, vector_in_dims)
    batch_size = _manual_vector_batch_size(settings)
    results = []

    for start in range(0, vector_count, batch_size):
        stop = min(start + batch_size, vector_count)
        vector = _vector_tree_slice(vector_tree, vector_in_dims, start, stop)
        result = runner(vector)
        results.append(result)

    return _cat_tensor_trees(tuple(results), 0)


def _run_vector_vmap(
    execution: StandardExecution,
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    return _run_tensor_tree_vector_vmap(
        execution.vector,
        execution.candidate.settings,
        runner,
    )


def _run_tensor_tree_by_vectorization_mode(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    mode = settings.get("vectorization.mode")

    if mode == "single_loop":
        return _run_tensor_tree_vector_single_loop(vector_tree, settings, runner)

    if mode == "manual_batch":
        return _run_tensor_tree_vector_manual_batches(vector_tree, settings, runner)

    if mode == "vmap":
        return _run_tensor_tree_vector_vmap(vector_tree, settings, runner)

    return runner(vector_tree)


def _run_tensor_tree_vector_vmap(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    vector_in_dims = _vector_tree_in_dims(
        vector_tree,
        settings,
    )

    return _torch_func_vmap(
        runner,
        in_dims=(vector_in_dims,),
        randomness=settings["vectorization.randomness"],
        chunk_size=_vmap_chunk_size(settings),
    )(vector_tree)


def _run_vector_single_loop(
    execution: StandardExecution,
    runner: Callable[[StandardExecution], TensorTree],
) -> TensorTree:
    def vector_runner(vector: TensorTree) -> TensorTree:
        return runner(_execution_with_vector(execution, vector))

    return _run_tensor_tree_vector_single_loop(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_tensor_tree_vector_single_loop(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    def vector_runner(vector: TensorTree, _: bool) -> TensorTree:
        return runner(vector)

    return _run_tensor_tree_vector_single_loop_with_last(
        vector_tree,
        settings,
        vector_runner,
    )


def _run_tensor_tree_vector_single_loop_with_last(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree, bool], TensorTree],
) -> TensorTree:
    vector_in_dims = _vector_tree_in_dims(
        vector_tree,
        settings,
    )
    vector_count = _vector_tree_batch_size(vector_tree, vector_in_dims)
    results = []

    for index in range(vector_count):
        vector = _vector_tree_select(vector_tree, vector_in_dims, index)
        result = runner(vector, index == vector_count - 1)
        results.append(result)

    return _stack_tensor_trees(tuple(results), 0)


def _hvp_uses_reverse_reuse(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("hvp.graph_schedule") == "retain_graph_across_vectors"
        or settings.get("hvp.primal_reuse") == "reuse_primal"
    )


def _run_hvp_reused_reverse_vectors(execution: StandardExecution) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)
    active_params = _grad_enabled_params(execution.params)
    value = scalar_function(active_params)
    retain_gradient_graph = (
        execution.candidate.settings.get("hvp.graph_schedule")
        == "retain_graph_across_vectors"
    )

    if retain_gradient_graph:
        return _run_hvp_reused_gradient_graph_vectors(
            execution,
            active_params,
            value,
        )

    return _run_hvp_reused_primal_vectors(
        execution,
        active_params,
        value,
    )


def _run_hvp_reused_gradient_graph_vectors(
    execution: StandardExecution,
    active_params: ParameterTree,
    value: torch.Tensor,
) -> TensorTree:
    gradient_tree = _hvp_gradient_tree(
        active_params,
        value,
        retain_graph=True,
    )

    def vector_runner(vector: TensorTree, is_last: bool) -> TensorTree:
        return _hvp_from_gradient_tree(
            active_params,
            gradient_tree,
            vector,
            execution.candidate.settings,
            retain_graph=not is_last,
        )

    return _run_tensor_tree_vector_single_loop_with_last(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_hvp_reused_primal_vectors(
    execution: StandardExecution,
    active_params: ParameterTree,
    value: torch.Tensor,
) -> TensorTree:
    def vector_runner(vector: TensorTree, is_last: bool) -> TensorTree:
        gradient_tree = _hvp_gradient_tree(
            active_params,
            value,
            retain_graph=not is_last,
        )

        return _hvp_from_gradient_tree(
            active_params,
            gradient_tree,
            vector,
            execution.candidate.settings,
            retain_graph=False,
        )

    return _run_tensor_tree_vector_single_loop_with_last(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _hvp_gradient_tree(
    active_params: ParameterTree,
    value: torch.Tensor,
    *,
    retain_graph: bool,
) -> TensorTree:
    leaves = tuple(active_params.values())
    gradient_leaves = torch.autograd.grad(
        value,
        leaves,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient
            for leaf, gradient in zip(leaves, gradient_leaves, strict=True)
        ),
    )


def _hvp_from_gradient_tree(
    active_params: ParameterTree,
    gradient_tree: TensorTree,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    retain_graph: bool,
) -> TensorTree:
    leaves = tuple(active_params.values())
    dot = _tree_dot_runtime(settings, gradient_tree, vector)
    hvp_leaves = torch.autograd.grad(
        dot,
        leaves,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(leaf) if hvp is None else hvp.detach()
            for leaf, hvp in zip(leaves, hvp_leaves, strict=True)
        ),
    )


def _run_hvp_vector_vmap(execution: StandardExecution) -> TensorTree:
    if execution.path not in HVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires linearize_grad HVP"
        raise MaterializationError(message)

    if execution.linearized_hvp is not None:
        hvp_function = execution.linearized_hvp
    else:
        gradient_function = torch.func.grad(_hvp_scalar_function(execution))
        _, hvp_function = torch.func.linearize(
            gradient_function,
            execution.params,
        )

    return _run_vector_vmap(execution, hvp_function)


def _hvp_scalar_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], torch.Tensor]:
    if execution.compiled_scalar_function is not None:
        return execution.compiled_scalar_function

    if _uses_stateful_module_call(execution):
        return _stateful_module_scalar_function(execution)

    scalar = _scalar_objective(execution.operator, execution.scalar_objectives)

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        settings = execution.candidate.settings

        return scalar(
            _model_compute_tree(active_params, settings),
            _model_compute_tree(execution.buffers, settings),
            _model_compute_batch(execution.batch, settings),
            execution.context,
        )

    return scalar_function


def _run_hvp_forward_ad_path(
    execution: StandardExecution,
) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)

    def gradient_function(active_params: ParameterTree) -> TensorTree:
        active_leaves = tree_leaves(active_params)
        value = scalar_function(active_params)
        gradients = torch.autograd.grad(
            value,
            active_leaves,
            allow_unused=True,
            create_graph=True,
        )

        return tree_from_leaves(
            active_params,
            tuple(
                torch.zeros_like(leaf) if gradient is None else gradient
                for leaf, gradient in zip(active_leaves, gradients, strict=True)
            ),
        )

    primal_params = _grad_enabled_params(execution.params)
    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)

    with torch.autograd.forward_ad.dual_level():
        dual_params = {
            name: torch.autograd.forward_ad.make_dual(param, vector)
            for (name, param), vector in zip(
                primal_params.items(),
                vector_leaves,
                strict=True,
            )
        }
        dual_gradients = gradient_function(dual_params)

        def tangent_leaf(output: torch.Tensor) -> torch.Tensor:
            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)

            if tangent is None:
                return torch.zeros_like(primal)

            return tangent

        result = tree_map(tangent_leaf, dual_gradients)

    _require_finite_tree(result, "HVP result")

    return result


def _run_hvp_vhp_path(
    execution: StandardExecution,
) -> TensorTree:
    parameter_items = tuple(execution.params.items())
    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)
    compiled_scalar_function = _hvp_scalar_function(execution)

    def scalar_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }

        return compiled_scalar_function(active_params)

    _, result_leaves = torch.autograd.functional.vhp(
        scalar_function,
        parameter_leaves,
        vector_leaves,
    )

    return tree_from_leaves(execution.params, result_leaves)


def _run_ggnvp(execution: StandardExecution) -> TensorTree:
    if (
        execution.compiled_inner is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_full_product"
    ):
        return execution.compiled_inner()

    return _run_ggnvp_by_path(execution)


def _run_ggnvp_by_path(execution: StandardExecution) -> TensorTree:
    return _run_single_vectorized_by_path(
        execution,
        (
            GGN_DENSE_PATH,
            GGN_JVP_HESSIAN_VJP_PATH,
            GGN_FORWARD_AD_HESSIAN_VJP_PATH,
            GGN_LINEARIZE_HESSIAN_VJP_PATH,
        ),
        _run_ggnvp_single_vector,
        _run_ggnvp_single_loop_vector,
        _run_ggnvp_vector_vmap,
    )


def _run_ggnvp_single_vector(execution: StandardExecution) -> TensorTree:
    if execution.path in {
        GGN_JVP_HESSIAN_VJP_PATH,
        GGN_FORWARD_AD_HESSIAN_VJP_PATH,
        GGN_LINEARIZE_HESSIAN_VJP_PATH,
    }:
        return _run_ggnvp_jvp_hessian_vjp(execution)

    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)
    vector_tensor = _parameter_order_vector(execution)

    def tensor_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }
        output = _call_function_objective(execution, function, active_params)

        if not isinstance(output, torch.Tensor):
            message = "dense GGNVP requires tensor function output"
            raise MaterializationError(message)

        return output.reshape(-1)

    output = tensor_function(*parameter_leaves)
    _require_finite_tensor(vector_tensor, "GGN vector")
    ggn_batch_size = _ggn_batch_size(execution.candidate.settings)

    if ggn_batch_size is not None:
        result = _dense_ggnvp_batched_rows(
            execution,
            tensor_function,
            parameter_leaves,
            vector_tensor,
            ggn_batch_size,
        )

        return _wrap_flat_vector(execution.params, result)

    jacobian = _dense_jacobian_tree(
        tensor_function,
        parameter_leaves,
        output.numel(),
    )
    _require_finite_tensor(jacobian, "GGN jacobian")
    output_vector = _parameter_blocked_matrix_vector_product(
        jacobian,
        vector_tensor.reshape(-1),
        execution.candidate.settings,
        execution.parameter_surface,
    )
    output_cotangent = _ggn_loss_hessian_product_by_path(
        execution,
        output,
        _wrap_flat_vector(output, output_vector),
    )
    loss_vector = _flatten_vector(output_cotangent)
    result = _jacobian_transpose_product(
        jacobian,
        loss_vector,
        execution.candidate.settings,
        execution.parameter_surface,
    )
    _require_finite_tensor(result, "GGN result")

    return _wrap_flat_vector(execution.params, result)


def _dense_ggnvp_batched_rows(
    execution: StandardExecution,
    tensor_function: Callable[..., torch.Tensor],
    parameter_leaves: tuple[torch.Tensor, ...],
    vector_tensor: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    settings = execution.candidate.settings
    active_leaves = tuple(
        parameter.detach().clone().requires_grad_(True)
        for parameter in parameter_leaves
    )
    output = tensor_function(*active_leaves)
    output_width = output.numel()
    parameter_width = vector_tensor.numel()
    output_vector = torch.empty(
        output_width,
        dtype=output.dtype,
        device=output.device,
    )

    for start in range(0, output_width, batch_size):
        stop = min(start + batch_size, output_width)
        jacobian_block = _dense_jacobian_row_block(
            output,
            active_leaves,
            parameter_width,
            start,
            stop,
        )
        output_vector[start:stop] = _parameter_blocked_matrix_vector_product(
            jacobian_block,
            vector_tensor.reshape(-1),
            settings,
            execution.parameter_surface,
        )

    output_cotangent = _ggn_loss_hessian_product_by_path(
        execution,
        output,
        _wrap_flat_vector(output, output_vector),
    )
    loss_vector = _flatten_vector(output_cotangent)
    result = torch.zeros(
        parameter_width,
        dtype=loss_vector.dtype,
        device=loss_vector.device,
    )

    for start in range(0, output_width, batch_size):
        stop = min(start + batch_size, output_width)
        jacobian_block = _dense_jacobian_row_block(
            output,
            active_leaves,
            parameter_width,
            start,
            stop,
        )
        result = result + _jacobian_transpose_product(
            jacobian_block,
            loss_vector[start:stop],
            settings,
            execution.parameter_surface,
        )

    _require_finite_tensor(result, "GGN batched result")

    return result


def _dense_jacobian_row_block(
    output: torch.Tensor,
    parameter_leaves: tuple[torch.Tensor, ...],
    parameter_width: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    flat_output = output.reshape(-1)
    rows = []

    for index in range(start, stop):
        gradients = torch.autograd.grad(
            flat_output[index],
            parameter_leaves,
            retain_graph=True,
            allow_unused=True,
        )
        row_parts = tuple(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(
                parameter_leaves,
                gradients,
                strict=True,
            )
        )
        rows.append(torch.cat(row_parts))

    if not rows:
        return torch.empty(
            0,
            parameter_width,
            dtype=output.dtype,
            device=output.device,
        )

    return torch.stack(tuple(rows))


def _run_ggnvp_single_loop_vector(execution: StandardExecution) -> TensorTree:
    return _run_ggnvp_single_vector(
        dataclasses.replace(
            execution,
            compiled_inner=None,
            compiled_ggn_loss_product=None,
            compiled_ggn_vjp=None,
        )
    )


def _run_ggnvp_vector_vmap(execution: StandardExecution) -> TensorTree:
    if execution.path not in GGN_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires a torch.func GGN JVP path"
        raise MaterializationError(message)

    if execution.candidate.settings.get("ggn.vjp_path") != "torch_func_vjp":
        message = "vectorization.mode=vmap requires ggn.vjp_path=torch_func_vjp"
        raise MaterializationError(message)

    tensor_function = _ggn_tensor_function(execution)

    if execution.path == GGN_LINEARIZE_HESSIAN_VJP_PATH:
        output, jvp_function = torch.func.linearize(tensor_function, execution.params)
    else:
        output = tensor_function(execution.params)

        def jvp_function(vector: TensorTree) -> TensorTree:
            return jvp_anchor(tensor_function, execution.params, vector)

    _require_finite_tree(output, "GGN output")
    _require_ggn_loss_hessian_vector_vmap_inputs(execution, output)
    pullback = _vjp_pullback(tensor_function, execution.params)

    def ggn_function(vector: TensorTree) -> TensorTree:
        output_jvp = jvp_function(vector)
        output_jvp = _runtime_intermediate_residency_tree(
            output_jvp,
            execution.candidate.settings,
            execution.intermediate_transform,
        )
        output_cotangent = _ggn_loss_hessian_product_unchecked(
            execution,
            output,
            output_jvp,
        )
        output_cotangent = _runtime_intermediate_residency_tree(
            output_cotangent,
            execution.candidate.settings,
            execution.intermediate_transform,
        )
        (result,) = pullback(output_cotangent)

        return result

    result = _run_vector_vmap(execution, ggn_function)
    _require_finite_tree(result, "GGN result")

    return result


def _require_ggn_loss_hessian_vector_vmap_inputs(
    execution: StandardExecution,
    output: TensorTree,
) -> None:
    if (
        execution.candidate.settings.get("ggn.loss_hessian_path")
        == "closed_form_softmax_ce_kl"
    ):
        return

    flat_output = _flatten_vector(output)
    loss_hessian = _batch_tensor(execution.batch, "loss_hessian")
    _require_loss_hessian_shape(loss_hessian, flat_output.numel())
    _require_finite_tensor(loss_hessian, "loss_hessian")


def _run_ggnvp_jvp_hessian_vjp(execution: StandardExecution) -> TensorTree:
    tensor_function = _ggn_tensor_function(execution)
    output, output_jvp = _ggn_output_and_jvp(execution, tensor_function)
    output_jvp = _runtime_intermediate_residency_tree(
        output_jvp,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    if _ggn_recomputes_jvp(execution.candidate.settings):
        _, output_jvp = _ggn_output_and_jvp(execution, tensor_function)
        output_jvp = _runtime_intermediate_residency_tree(
            output_jvp,
            execution.candidate.settings,
            execution.intermediate_transform,
        )

    output_cotangent = _ggn_loss_hessian_product(
        execution,
        output,
        output_jvp,
    )
    output_cotangent = _runtime_intermediate_residency_tree(
        output_cotangent,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    if _ggn_recomputes_output_cotangent(execution.candidate.settings):
        output_cotangent = _ggn_loss_hessian_product(
            execution,
            output,
            output_jvp,
        )
        output_cotangent = _runtime_intermediate_residency_tree(
            output_cotangent,
            execution.candidate.settings,
            execution.intermediate_transform,
        )

    _require_finite_tree(output_cotangent, "GGN output cotangent")

    result = _run_ggnvp_vjp(execution, tensor_function, output_cotangent)
    _require_finite_tree(result, "GGN result")

    return result


def _ggn_tensor_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    function = _function_objective(execution.operator, execution.function_objectives)

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return _call_function_objective(execution, function, active_params)

    return tensor_function


def _ggn_loss_product_warm_inputs(
    execution: StandardExecution,
) -> tuple[TensorTree, TensorTree]:
    tensor_function = _ggn_tensor_function(execution)
    output, output_jvp = _ggn_output_and_jvp_by_path(execution, tensor_function)
    output_jvp = _runtime_intermediate_residency_tree(
        output_jvp,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    return output, output_jvp


def _ggn_vjp_warm_input(execution: StandardExecution) -> TensorTree:
    output, output_jvp = _ggn_loss_product_warm_inputs(execution)

    output_cotangent = _ggn_loss_hessian_product_by_path(execution, output, output_jvp)

    return _runtime_intermediate_residency_tree(
        output_cotangent,
        execution.candidate.settings,
        execution.intermediate_transform,
    )


def _ggn_output_and_jvp(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> tuple[TensorTree, TensorTree]:
    if (
        execution.compiled_ggn_jvp is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_jvp"
    ):
        return execution.compiled_ggn_jvp()

    return _ggn_output_and_jvp_by_path(execution, tensor_function)


def _ggn_output_and_jvp_by_path(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> tuple[TensorTree, TensorTree]:
    if execution.path == GGN_LINEARIZE_HESSIAN_VJP_PATH:
        output, jvp_function = torch.func.linearize(tensor_function, execution.params)

        return output, jvp_function(execution.vector)

    if execution.path == GGN_FORWARD_AD_HESSIAN_VJP_PATH:
        return _forward_ad_output_and_jvp(
            tensor_function,
            execution.params,
            execution.vector,
        )

    jvp_result = torch.func.jvp(
        tensor_function,
        (execution.params,),
        (execution.vector,),
    )

    return jvp_result[0], jvp_result[1]


def _forward_ad_output_and_jvp(
    function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    vector: TensorTree,
) -> tuple[TensorTree, TensorTree]:
    vector_params = _parameter_tree_from_tensor_tree(vector, "GGN vector")

    with torch.autograd.forward_ad.dual_level():
        dual_params = {
            name: torch.autograd.forward_ad.make_dual(params[name], vector_params[name])
            for name in params
        }
        dual_output = function(dual_params)

        return _split_forward_ad_dual_tree(dual_output)


def _split_forward_ad_dual_tree(tree: TensorTree) -> tuple[TensorTree, TensorTree]:
    if isinstance(tree, torch.Tensor):
        primal, tangent = torch.autograd.forward_ad.unpack_dual(tree)

        if tangent is None:
            return primal, torch.zeros_like(primal)

        return primal, tangent

    if isinstance(tree, tuple):
        pairs = tuple(_split_forward_ad_dual_tree(child) for child in tree)

        return (
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
        )

    if isinstance(tree, dict):
        pairs = {key: _split_forward_ad_dual_tree(value) for key, value in tree.items()}

        return (
            {key: pair[0] for key, pair in pairs.items()},
            {key: pair[1] for key, pair in pairs.items()},
        )

    message = "forward AD output must be a tensor tree"
    raise MaterializationError(message)


def _ggn_recomputes_jvp(settings: Mapping[str, Any]) -> bool:
    return settings.get("ggn.jvp_reuse") == "recompute_jvp"


def _ggn_recomputes_output_cotangent(settings: Mapping[str, Any]) -> bool:
    return settings.get("ggn.cotangent_reuse") == "recompute_output_cotangent"


def _ggn_loss_hessian_product(
    execution: StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if (
        execution.compiled_ggn_loss_product is not None
        and execution.candidate.settings.get("compile.boundary")
        == "ggn_loss_hessian_product"
    ):
        return execution.compiled_ggn_loss_product(output, output_jvp)

    return _ggn_loss_hessian_product_by_path(execution, output, output_jvp)


def _ggn_loss_hessian_product_by_path(
    execution: StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.loss_hessian_path")

    if path == "closed_form_softmax_ce_kl":
        return _ggn_closed_form_softmax_ce_kl_product(
            execution.candidate.settings,
            output,
            output_jvp,
        )

    if path == "autodiff_loss_hvp":
        return _ggn_autodiff_loss_hvp(execution, output, output_jvp, validate=True)

    message = f"ggn.loss_hessian_path is unsupported: {path}"
    raise MaterializationError(message)


def _ggn_loss_hessian_product_unchecked(
    execution: StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.loss_hessian_path")

    if path == "closed_form_softmax_ce_kl":
        return _ggn_closed_form_softmax_ce_kl_product_unchecked(
            execution.candidate.settings,
            output,
            output_jvp,
        )

    if path == "autodiff_loss_hvp":
        return _ggn_autodiff_loss_hvp(execution, output, output_jvp, validate=False)

    message = f"ggn.loss_hessian_path is unsupported: {path}"
    raise MaterializationError(message)


def _ggn_autodiff_loss_hvp(
    execution: StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
    *,
    validate: bool,
) -> TensorTree:
    flat_output = _flatten_vector(output)
    flat_output_jvp = _flatten_vector(output_jvp)
    loss_hessian = _batch_tensor(execution.batch, "loss_hessian")

    if validate:
        _require_loss_hessian_shape(loss_hessian, flat_output_jvp.numel())
        _require_finite_tensor(loss_hessian, "loss_hessian")
        _require_finite_tensor(flat_output_jvp, "GGN output JVP")

    def output_loss(flat_value: torch.Tensor) -> torch.Tensor:
        hessian_value = _matmul_runtime(
            execution.candidate.settings,
            loss_hessian,
            flat_value,
        )

        return 0.5 * _dot_runtime(
            execution.candidate.settings,
            flat_value,
            hessian_value,
        )

    product = torch.func.jvp(
        torch.func.grad(output_loss),
        (flat_output,),
        (flat_output_jvp,),
    )[1]

    return _wrap_flat_vector(
        output,
        product,
    )


def _ggn_closed_form_softmax_ce_kl_product(
    settings: Mapping[str, Any],
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if not isinstance(output, torch.Tensor) or not isinstance(output_jvp, torch.Tensor):
        message = "closed-form CE/KL GGN requires tensor logits and tensor JVP"
        raise MaterializationError(message)

    if output.shape != output_jvp.shape:
        message = "closed-form CE/KL logits and JVP shapes must match"
        raise MaterializationError(message)

    if output.ndim == 0:
        message = "closed-form CE/KL logits must have a class dimension"
        raise MaterializationError(message)

    _require_finite_tensor(output, "GGN logits")
    _require_finite_tensor(output_jvp, "GGN output JVP")
    token_block_size = _token_block_size(settings)

    if token_block_size is not None:
        return _softmax_ce_kl_product_token_loop(
            settings,
            output,
            output_jvp,
            token_block_size,
        )

    return _softmax_ce_kl_product_without_token_loop(settings, output, output_jvp)


def _ggn_closed_form_softmax_ce_kl_product_unchecked(
    settings: Mapping[str, Any],
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if not isinstance(output, torch.Tensor) or not isinstance(output_jvp, torch.Tensor):
        message = "closed-form CE/KL GGN requires tensor logits and tensor JVP"
        raise MaterializationError(message)

    if output.shape != output_jvp.shape:
        message = "closed-form CE/KL logits and JVP shapes must match"
        raise MaterializationError(message)

    if output.ndim == 0:
        message = "closed-form CE/KL logits must have a class dimension"
        raise MaterializationError(message)

    token_block_size = _token_block_size(settings)

    if token_block_size is not None:
        return _softmax_ce_kl_product_token_loop(
            settings,
            output,
            output_jvp,
            token_block_size,
        )

    return _softmax_ce_kl_product_without_token_loop(settings, output, output_jvp)


def _softmax_ce_kl_product_token_loop(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    class_count = logits.shape[-1]
    flat_logits = logits.reshape(-1, class_count)
    flat_tangent = tangent.reshape(-1, class_count)
    outputs = []

    for start in range(0, flat_logits.shape[0], block_size):
        stop = min(start + block_size, flat_logits.shape[0])
        logits_block = flat_logits[start:stop]
        tangent_block = flat_tangent[start:stop]
        outputs.append(
            _softmax_ce_kl_product_without_token_loop(
                settings,
                logits_block,
                tangent_block,
            )
        )

    return torch.cat(tuple(outputs), dim=0).reshape_as(logits)


def _softmax_ce_kl_product_without_token_loop(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    kernel = settings.get("ggn.loss_hessian_kernel")

    if kernel == "dense_global":
        return _softmax_ce_kl_product_dense_global(settings, logits, tangent)

    if kernel == "streaming_global":
        return _softmax_ce_kl_product_streaming_global(settings, logits, tangent)

    if kernel == "two_pass_chunked_global":
        block_size = _class_block_size_with_exact_global_normalization(settings)

        return _softmax_ce_kl_product_two_pass_chunked_global(
            settings,
            logits,
            tangent,
            block_size,
        )

    message = f"ggn.loss_hessian_kernel is unsupported: {kernel}"
    raise MaterializationError(message)


def _softmax_ce_kl_product_dense_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    runtime_probabilities = _accumulation_tensor(probabilities, settings)
    runtime_tangent = _accumulation_tensor(tangent, settings)
    mean_tangent = (runtime_probabilities * runtime_tangent).sum(
        dim=-1,
        keepdim=True,
    )

    return runtime_probabilities * (runtime_tangent - mean_tangent)


def _softmax_ce_kl_product_streaming_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    unnormalized = _accumulation_tensor((logits - max_logits).exp(), settings)
    runtime_tangent = _accumulation_tensor(tangent, settings)
    denominator = unnormalized.sum(dim=-1, keepdim=True)
    probabilities = unnormalized / denominator
    mean_tangent = (probabilities * runtime_tangent).sum(dim=-1, keepdim=True)

    return probabilities * (runtime_tangent - mean_tangent)


def _softmax_ce_kl_product_two_pass_chunked_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    chunks = torch.split(logits, block_size, dim=-1)
    tangent_chunks = torch.split(tangent, block_size, dim=-1)
    max_logits = chunks[0].max(dim=-1, keepdim=True).values

    for chunk in chunks[1:]:
        max_logits = torch.maximum(max_logits, chunk.max(dim=-1, keepdim=True).values)

    accumulator_template = _accumulation_tensor(max_logits, settings)
    denominator = torch.zeros_like(accumulator_template)
    weighted_tangent_sum = torch.zeros_like(accumulator_template)

    for chunk, tangent_chunk in zip(chunks, tangent_chunks, strict=True):
        unnormalized = _accumulation_tensor((chunk - max_logits).exp(), settings)
        runtime_tangent_chunk = _accumulation_tensor(tangent_chunk, settings)
        denominator = denominator + unnormalized.sum(dim=-1, keepdim=True)
        weighted_tangent_sum = weighted_tangent_sum + (
            unnormalized * runtime_tangent_chunk
        ).sum(dim=-1, keepdim=True)

    mean_tangent = weighted_tangent_sum / denominator
    outputs = []

    for chunk, tangent_chunk in zip(chunks, tangent_chunks, strict=True):
        unnormalized = _accumulation_tensor((chunk - max_logits).exp(), settings)
        runtime_tangent_chunk = _accumulation_tensor(tangent_chunk, settings)
        probabilities = unnormalized / denominator
        outputs.append(probabilities * (runtime_tangent_chunk - mean_tangent))

    return torch.cat(tuple(outputs), dim=-1)


def _class_block_size_with_exact_global_normalization(
    settings: Mapping[str, Any],
) -> int:
    key = "chunk.class_block_size_with_exact_global_normalization"
    value = settings.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _run_ggnvp_vjp(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    block_size = _output_cotangent_block_size(execution.candidate.settings)

    if block_size is not None:

        def run_block(block: TensorTree) -> TensorTree:
            return _run_ggnvp_vjp_single_cotangent(execution, tensor_function, block)

        return _run_output_cotangent_blocks(
            execution,
            output_cotangent,
            block_size,
            run_block,
        )

    return _run_ggnvp_vjp_single_cotangent(
        execution,
        tensor_function,
        output_cotangent,
    )


def _run_ggnvp_vjp_single_cotangent(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    if (
        execution.compiled_ggn_vjp is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_vjp"
    ):
        return execution.compiled_ggn_vjp(output_cotangent)

    return _run_ggnvp_vjp_by_path(execution, tensor_function, output_cotangent)


def _run_output_cotangent_blocks(
    execution: StandardExecution,
    output_cotangent: TensorTree,
    block_size: int,
    run_block: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    blocks = _cotangent_blocks(output_cotangent, block_size)
    result = run_block(blocks[0])

    for block in blocks[1:]:
        result = _tree_add_runtime(
            execution.candidate.settings,
            result,
            run_block(block),
        )

    return result


def _cotangent_blocks(
    output_cotangent: TensorTree,
    block_size: int,
) -> tuple[TensorTree, ...]:
    flat = _flatten_vector(output_cotangent)
    blocks = []

    for start in range(0, flat.numel(), block_size):
        stop = min(start + block_size, flat.numel())
        blocks.append(_flat_cotangent_block(output_cotangent, start, stop))

    return tuple(blocks)


def _flat_cotangent_block(
    output_cotangent: TensorTree,
    start: int,
    stop: int,
) -> TensorTree:
    leaves = []
    offset = 0

    for leaf in tree_leaves(output_cotangent):
        width = leaf.numel()
        leaf_stop = offset + width
        block = torch.zeros_like(leaf).reshape(-1)
        local_start = max(start, offset)
        local_stop = min(stop, leaf_stop)

        if local_start < local_stop:
            source = leaf.reshape(-1)
            block[local_start - offset : local_stop - offset] = source[
                local_start - offset : local_stop - offset
            ]

        leaves.append(block.reshape_as(leaf))
        offset = leaf_stop

    return tree_from_leaves(output_cotangent, tuple(leaves))


def _run_ggnvp_vjp_by_path(
    execution: StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.vjp_path")

    if path == "torch_func_vjp":
        pullback = _vjp_pullback(tensor_function, execution.params)
        (result,) = pullback(output_cotangent)

        return result

    if path == "autograd_grad_outputs":
        return _autograd_grad_outputs_vjp(
            tensor_function,
            execution.params,
            output_cotangent,
        )

    message = f"ggn.vjp_path is unsupported: {path}"
    raise MaterializationError(message)


def _dense_jacobian_tree(
    function: Callable[..., torch.Tensor],
    parameter_leaves: tuple[torch.Tensor, ...],
    output_numel: int,
) -> torch.Tensor:
    jacobian = torch.autograd.functional.jacobian(function, parameter_leaves)
    jacobian_leaves = (jacobian,) if isinstance(jacobian, torch.Tensor) else jacobian
    parts = tuple(
        part.reshape(output_numel, parameter.numel())
        for part, parameter in zip(jacobian_leaves, parameter_leaves, strict=True)
    )

    return torch.cat(parts, dim=1)


def _require_loss_hessian_shape(
    loss_hessian: torch.Tensor,
    output_numel: int,
) -> None:
    if loss_hessian.shape != (output_numel, output_numel):
        message = "loss_hessian shape must match flattened function output"
        raise MaterializationError(message)


def _run_fisher_vp(execution: StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(execution, FISHER_FAMILY_RUNTIME_ROWS["fisher_vp"])


def _run_fisher_family_vp(
    execution: StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    def single_vector(row_execution: StandardExecution) -> TensorTree:
        row["precheck"](row_execution)

        return _run_fisher_family_vp_single_vector(row_execution, row)

    def vmap(row_execution: StandardExecution) -> TensorTree:
        row["precheck"](row_execution)

        return _run_fisher_family_vp_vector_vmap(row_execution, row)

    return _run_single_vectorized_by_path(
        execution,
        row["paths"],
        single_vector,
        single_vector,
        vmap,
    )


def _run_fisher_family_vp_single_vector(
    execution: StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    if execution.path in row["streaming_paths"]:
        row["require_streaming"](execution)
        result = _streaming_score_gradient_product(
            execution,
            row["streaming_normalization"](execution),
            row["streaming_label"],
        )
        row["check_result"](execution, result)

        return _wrap_flat_vector(execution.params, result)

    if execution.path == row["blockwise_path"]:
        row["require_matrix"](execution)

        return _run_blockwise_score_matrix_product(
            execution,
            row["block_batch_key"],
            row["blockwise_normalization"](execution),
            row["matrix_batch_key"],
        )

    row["require_matrix"](execution)
    score_matrix = _batch_tensor(execution.batch, row["matrix_batch_key"])
    score_matrix = _loss_scaled_score_matrix(execution, score_matrix)

    return _run_score_matrix_product_single_vector(
        execution,
        score_matrix,
        row["matrix_normalization"](execution, score_matrix),
        row["vector_label"],
        row["result_label"],
        row["check_result"],
    )


def _run_fisher_family_vp_vector_vmap(
    execution: StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    if execution.path == row["blockwise_path"]:
        row["require_matrix"](execution)
        result = _blockwise_score_matrix_product_batch_vmap(
            execution,
            row["block_batch_key"],
            row["blockwise_normalization"](execution),
            row["blockwise_label"],
        )
        row["check_result"](execution, result)

        return _wrap_flat_vector_batch(execution.params, result)

    if execution.path in row["streaming_paths"]:
        row["require_streaming"](execution)

        return _run_streaming_score_gradient_product_vmap(
            execution,
            row["streaming_normalization"](execution),
            row["streaming_label"],
        )

    score_matrix = _score_fisher_matrix_for_product(
        execution,
        row["streaming_paths"],
        row["dense_path"],
        row["matrix_batch_key"],
        row["streaming_matrix"],
        row["require_streaming"],
        row["require_matrix"],
        row["vmap_error_message"],
    )
    result = _score_matrix_product_batch_vmap(
        execution,
        score_matrix,
        row["matrix_normalization"](execution, score_matrix),
        row["streaming_label"],
    )
    row["check_result"](execution, result)

    return _wrap_flat_vector_batch(execution.params, result)


def _score_fisher_matrix_for_product(
    execution: StandardExecution,
    streaming_paths: tuple[str, ...],
    dense_path: str,
    dense_batch_key: str,
    streaming_matrix: Callable[[StandardExecution], torch.Tensor],
    require_streaming: Callable[[StandardExecution], None],
    require_dense: Callable[[StandardExecution], None],
    error_message: str,
) -> torch.Tensor:
    if execution.path in streaming_paths:
        require_streaming(execution)

        return _loss_scaled_score_matrix(execution, streaming_matrix(execution))

    if execution.path == dense_path:
        require_dense(execution)

        return _loss_scaled_score_matrix(
            execution,
            _batch_tensor(execution.batch, dense_batch_key),
        )

    raise MaterializationError(error_message)


def _require_explicit_score_fisher_execution(execution: StandardExecution) -> None:
    _require_explicit_score_fisher_semantics(execution.operator)


def _require_valid_fisher_execution(execution: StandardExecution) -> None:
    _require_valid_fisher_semantics(execution.operator)


def _fisher_score_matrix_normalization(
    execution: StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    _ = score_matrix

    return _fisher_normalization(execution)


def _skip_score_fisher_requirement(execution: StandardExecution) -> None:
    _ = execution


def _fisher_score_gradients(execution: StandardExecution) -> torch.Tensor:
    return _score_gradient_matrix_from_builders(
        execution,
        "fisher_score_grad",
        FISHER_SCORE_GRADIENT_BUILDERS,
        "fisher score-gradient boundary requires a score-gradient path",
    )


def _run_sampled_fisher_vp(execution: StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(
        execution,
        FISHER_FAMILY_RUNTIME_ROWS["sampled_fisher_vp"],
    )


def _sampled_fisher_score_gradients(execution: StandardExecution) -> torch.Tensor:
    return _score_gradient_matrix_from_builders(
        execution,
        "sampled_fisher_score_grad",
        SAMPLED_FISHER_SCORE_GRADIENT_BUILDERS,
        "sampled Fisher score-gradient boundary requires a score-gradient path",
    )


def _sampled_fisher_score_matrix_normalization(
    execution: StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    _ = score_matrix

    return _sampled_fisher_normalization(execution)


def _run_empirical_fisher_vp(execution: StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(
        execution,
        FISHER_FAMILY_RUNTIME_ROWS["empirical_fisher_vp"],
    )


def _empirical_fisher_gradients(execution: StandardExecution) -> torch.Tensor:
    return _score_gradient_matrix_from_builders(
        execution,
        "empirical_fisher_example_grad",
        EMPIRICAL_FISHER_GRADIENT_BUILDERS,
        "empirical Fisher example-gradient boundary requires a gradient path",
    )


def _score_gradient_matrix_from_builders(
    execution: StandardExecution,
    compile_boundary: str,
    builders: Mapping[str, Callable[[StandardExecution], torch.Tensor]],
    error_message: str,
    *,
    use_compiled: bool = True,
) -> torch.Tensor:
    if (
        use_compiled
        and execution.compiled_score_matrix is not None
        and execution.candidate.settings.get("compile.boundary") == compile_boundary
    ):
        return execution.compiled_score_matrix()

    if _uses_manual_per_example_schedule(execution):
        return _per_example_gradient_matrix_manual_batches(execution)

    return _per_example_gradient_matrix_from_builders(
        execution,
        builders,
        error_message,
    )


def _empirical_fisher_blockwise_normalization(
    execution: StandardExecution,
) -> float:
    blocks = _batch_tensor_blocks(
        execution.batch,
        "per_example_gradient_blocks",
    )

    return _empirical_fisher_normalization(
        execution.batch,
        execution.operator,
        blocks[0],
    )


def _empirical_fisher_score_matrix_normalization(
    execution: StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    return _empirical_fisher_normalization(
        execution.batch,
        execution.operator,
        score_matrix,
    )


def _run_per_example_gradient(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            PER_EXAMPLE_GRADIENT_LOOP_PATH,
            PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
            PER_EXAMPLE_GRADIENT_BACKWARD_PATH,
            PER_EXAMPLE_GRADIENT_VMAP_PATH,
        ),
    )
    accumulation = execution.candidate.settings.get("per_example_gradient.accumulation")

    if accumulation == "stacked_leading_axis":
        if (
            execution.compiled_score_matrix is not None
            and execution.candidate.settings.get("compile.boundary")
            == "per_example_gradient"
        ):
            matrix = execution.compiled_score_matrix()
        else:
            matrix = _per_example_gradient_matrix_without_manual_batch(execution)
    elif accumulation == "blockwise_stacked":
        matrix = _per_example_gradient_matrix_blockwise(execution)
    else:
        message = (
            "per_example_gradient.accumulation is required for per_example_gradient"
        )
        raise MaterializationError(message)

    if matrix.ndim != MATRIX_DIMS:
        message = "per-example gradient output must be a matrix"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "per-example gradient output")

    return _wrap_flat_vector_batch(execution.params, matrix)


def _run_blockwise_score_matrix_product(
    execution: StandardExecution,
    batch_key: str,
    normalization: float,
    label: str,
) -> TensorTree:
    blocks = _loss_scaled_score_blocks(
        execution,
        _batch_tensor_blocks(execution.batch, batch_key),
    )
    vector_tensor = _parameter_order_vector(execution)
    result = _blockwise_score_matrix_product(
        blocks,
        vector_tensor,
        normalization,
        label,
        execution.candidate.settings,
    )

    return _wrap_flat_vector(execution.params, result)


def _blockwise_score_matrix_product(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    normalization: float,
    label: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    first_block = blocks[0]
    offset = first_block.shape[1]
    score_dot = _matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width

        if stop > vector.numel():
            message = f"{label} block columns exceed vector length"
            raise MaterializationError(message)

        score_dot = score_dot + _matmul_runtime(settings, block, vector[offset:stop])
        offset = stop

    if offset != vector.numel():
        message = f"{label} block columns must match vector length"
        raise MaterializationError(message)

    pieces = tuple(_matmul_runtime(settings, block.T, score_dot) for block in blocks)
    result = torch.cat(pieces) / normalization
    _require_finite_tensor(result, f"{label} blockwise result")

    return result


def _skip_score_matrix_result_check(
    execution: StandardExecution,
    result: torch.Tensor,
) -> None:
    _ = execution, result


def _run_score_matrix_product_single_vector(
    execution: StandardExecution,
    score_gradients: torch.Tensor,
    normalization: float,
    vector_label: str,
    result_label: str,
    check_result: Callable[[StandardExecution, torch.Tensor], None],
) -> TensorTree:
    vector_tensor = _parameter_order_vector(execution)
    _require_finite_tensor(score_gradients, "score_gradients")
    _require_finite_tensor(vector_tensor, vector_label)
    result = _score_matrix_product(
        score_gradients,
        vector_tensor,
        normalization,
        execution.candidate.settings,
        execution.parameter_surface,
    )
    _require_finite_tensor(result, result_label)
    check_result(execution, result)

    return _wrap_flat_vector(execution.params, result)


def _score_matrix_product(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> torch.Tensor:
    _require_score_matrix_vector_shape(score_gradients, vector, "score_gradients")
    ranges = _parameter_column_ranges(vector.numel(), settings, parameter_surface)

    if ranges is not None:
        return _parameter_blocked_score_matrix_product(
            score_gradients,
            vector,
            normalization,
            settings,
            ranges,
        )

    score_dot = _matmul_runtime(settings, score_gradients, vector)

    return _matmul_runtime(settings, score_gradients.T, score_dot) / normalization


def _parameter_blocked_score_matrix_product(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
    ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    _require_score_matrix_vector_shape(score_gradients, vector, "score_gradients")
    score_dot = _parameter_blocked_matrix_vector_product(
        score_gradients,
        vector,
        settings,
        None,
        ranges,
    )
    result_chunks = []

    for start, stop in ranges:
        score_block = score_gradients[:, start:stop]
        result_chunks.append(_matmul_runtime(settings, score_block.T, score_dot))

    result = torch.cat(tuple(result_chunks)) / normalization
    _require_finite_tensor(result, "score matrix parameter-block result")

    return result


def _parameter_blocked_matrix_vector_product(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
    ranges: tuple[tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    column_ranges = (
        _parameter_column_ranges(vector.numel(), settings, parameter_surface)
        if ranges is None
        else ranges
    )

    if column_ranges is None:
        return _matmul_runtime(settings, matrix, vector)

    if matrix.ndim != MATRIX_DIMS:
        message = "parameter-block matrix must be two-dimensional"
        raise MaterializationError(message)

    if vector.ndim != 1:
        message = "parameter-block vector must be one-dimensional"
        raise MaterializationError(message)

    if matrix.shape[1] != vector.numel():
        message = "parameter-block matrix columns must match vector width"
        raise MaterializationError(message)

    chunks = []

    for start, stop in column_ranges:
        chunks.append(
            _matmul_runtime(settings, matrix[:, start:stop], vector[start:stop])
        )

    result = chunks[0]

    for chunk in chunks[1:]:
        result = result + chunk

    return result


def _jacobian_transpose_product(
    jacobian: torch.Tensor,
    cotangent: torch.Tensor,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> torch.Tensor:
    ranges = _parameter_column_ranges(jacobian.shape[1], settings, parameter_surface)

    if ranges is None:
        return _matmul_runtime(settings, jacobian.T, cotangent)

    if jacobian.ndim != MATRIX_DIMS:
        message = "GGN jacobian must be two-dimensional"
        raise MaterializationError(message)

    if cotangent.ndim != 1:
        message = "GGN cotangent must be one-dimensional"
        raise MaterializationError(message)

    if jacobian.shape[0] != cotangent.numel():
        message = "GGN jacobian rows must match cotangent width"
        raise MaterializationError(message)

    chunks = []

    for start, stop in ranges:
        chunks.append(_matmul_runtime(settings, jacobian[:, start:stop].T, cotangent))

    return torch.cat(tuple(chunks))


def _parameter_column_ranges(
    width: int,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> tuple[tuple[int, int], ...] | None:
    block_size = _parameter_block_size(settings)
    layer_block_size = _layer_block_size(settings)

    if block_size is not None and layer_block_size is not None:
        message = "parameter and layer chunk sizes cannot both be set"
        raise MaterializationError(message)

    if block_size is not None:
        return tuple(_parameter_block_ranges(width, block_size))

    if layer_block_size is None:
        return None

    return _layer_block_ranges(width, layer_block_size, parameter_surface)


def _layer_block_ranges(
    width: int,
    block_size: int,
    parameter_surface: ParameterSurface | None,
) -> tuple[tuple[int, int], ...]:
    if parameter_surface is None or not parameter_surface.layer_groups:
        message = "chunk.layer_block_size requires declared layer_groups"
        raise MaterializationError(message)

    group_ranges = _parameter_surface_group_ranges(
        parameter_surface,
        parameter_surface.layer_groups,
        "chunk.layer_block_size",
    )
    ranges = []

    for start in range(0, len(group_ranges), block_size):
        group_chunk = group_ranges[start : start + block_size]
        chunk_start = group_chunk[0][0]
        chunk_stop = group_chunk[-1][1]
        chunk_width = sum(stop - start for start, stop in group_chunk)

        if chunk_stop - chunk_start != chunk_width:
            message = "chunk.layer_block_size requires contiguous layer groups"
            raise MaterializationError(message)

        ranges.append((chunk_start, chunk_stop))

    if ranges[0][0] != 0 or ranges[-1][1] != width:
        message = "chunk.layer_block_size ranges must cover parameter width"
        raise MaterializationError(message)

    return tuple(ranges)


def _parameter_surface_group_ranges(
    parameter_surface: ParameterSurface,
    groups: tuple[tuple[str, ...], ...],
    key: str,
) -> tuple[tuple[int, int], ...]:
    offsets = {}
    offset = 0

    for name, shape in zip(
        parameter_surface.names,
        parameter_surface.shapes,
        strict=True,
    ):
        width = math.prod(shape)
        offsets[name] = (offset, offset + width)
        offset += width

    ranges = []

    for group in groups:
        group_ranges = tuple(offsets[name] for name in group)
        start = min(start for start, _ in group_ranges)
        stop = max(stop for _, stop in group_ranges)
        width = sum(stop - start for start, stop in group_ranges)

        if stop - start != width:
            message = f"{key} requires contiguous parameter groups"
            raise MaterializationError(message)

        ranges.append((start, stop))

    return tuple(ranges)


def _parameter_block_ranges(
    width: int,
    block_size: int,
) -> Iterator[tuple[int, int]]:
    for start in range(0, width, block_size):
        yield start, min(start + block_size, width)


def _require_score_matrix_vector_shape(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    label: str,
) -> None:
    if score_gradients.ndim != MATRIX_DIMS:
        message = f"{label} must be a two-dimensional tensor"
        raise MaterializationError(message)

    if vector.ndim != 1:
        message = "Fisher vector must flatten to a one-dimensional tensor"
        raise MaterializationError(message)

    if score_gradients.shape[0] == 0:
        message = f"{label} must have at least one row"
        raise MaterializationError(message)

    if score_gradients.shape[1] != vector.numel():
        message = f"{label} column count must match vector width"
        raise MaterializationError(message)


def _streaming_score_gradient_product(
    execution: StandardExecution,
    normalization: float,
    label: str,
) -> torch.Tensor:
    vector_tensor = _parameter_order_vector(execution)

    return _streaming_score_gradient_product_for_vector(
        execution,
        vector_tensor,
        normalization,
        label,
    )


def _streaming_score_gradient_product_for_vector(
    execution: StandardExecution,
    vector_tensor: torch.Tensor,
    normalization: float,
    label: str,
) -> torch.Tensor:

    if _uses_manual_per_example_schedule(execution):
        result = _streaming_score_gradient_product_manual_batches(
            execution,
            vector_tensor,
        )
    else:
        result = _streaming_score_gradient_product_without_manual_batch(
            execution,
            vector_tensor,
        )

    result = result / normalization
    _require_finite_tensor(result, f"{label} streaming result")

    return result


def _run_streaming_score_gradient_product_vmap(
    execution: StandardExecution,
    normalization: float,
    label: str,
) -> TensorTree:
    vector_batch = _flat_vector_batch(execution)
    chunk_size = _vmap_chunk_size(execution.candidate.settings)
    result = torch.zeros_like(vector_batch)

    for row in _streaming_gradient_rows(execution):
        result = _accumulate_streaming_gradient_row_batch(
            execution,
            result,
            row,
            vector_batch,
            chunk_size,
        )

    result = result / normalization
    _require_finite_tensor(result, f"{label} streaming batched result")

    return _wrap_flat_vector_batch(execution.params, result)


def _streaming_score_gradient_product_manual_batches(
    execution: StandardExecution,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    batch, batch_in_dims = _per_example_batch_in_dims(
        execution.batch,
        "per-example manual batching",
    )
    example_count = _per_example_batch_size(
        batch,
        batch_in_dims,
        "per-example manual batching",
    )
    batch_size = _per_example_manual_batch_size(execution)
    result = torch.zeros_like(vector_tensor)

    for start in range(0, example_count, batch_size):
        stop = min(start + batch_size, example_count)
        subbatch = _per_example_batch_slice(batch, batch_in_dims, start, stop)
        subexecution = dataclasses.replace(execution, batch=subbatch)
        result = result + _streaming_score_gradient_product_without_manual_batch(
            subexecution,
            vector_tensor,
        )

    return result


def _streaming_score_gradient_product_without_manual_batch(
    execution: StandardExecution,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(vector_tensor)

    for row in _streaming_gradient_rows_without_manual_batch(execution):
        result = _accumulate_streaming_gradient_row(
            execution,
            result,
            row,
            vector_tensor,
        )

    return result


def _streaming_gradient_rows(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    if _uses_manual_per_example_schedule(execution):
        batch, batch_in_dims = _per_example_batch_in_dims(
            execution.batch,
            "per-example manual batching",
        )
        example_count = _per_example_batch_size(
            batch,
            batch_in_dims,
            "per-example manual batching",
        )
        batch_size = _per_example_manual_batch_size(execution)

        for start in range(0, example_count, batch_size):
            stop = min(start + batch_size, example_count)
            subbatch = _per_example_batch_slice(batch, batch_in_dims, start, stop)
            subexecution = dataclasses.replace(execution, batch=subbatch)
            yield from _streaming_gradient_rows_without_manual_batch(subexecution)

        return

    yield from _streaming_gradient_rows_without_manual_batch(execution)


def _streaming_gradient_rows_without_manual_batch(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    compiled_rows = _compiled_streaming_gradient_rows(execution)

    if compiled_rows is not None:
        yield from compiled_rows

        return

    if execution.path in {
        FISHER_SCORE_GRADIENT_LOOP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
    }:
        yield from _streaming_gradient_rows_loop(execution)

        return

    if execution.path in {
        FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
    }:
        yield from _streaming_gradient_rows_torch_func(execution)

        return

    if execution.path in {
        FISHER_BACKWARD_MATERIALIZED_PATH,
        SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
        EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
    }:
        yield from _streaming_gradient_rows_backward(execution)

        return

    if execution.path in {
        FISHER_SCORE_GRADIENT_VMAP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
    }:
        yield from _streaming_gradient_rows_vmap(execution)

        return

    message = "streaming score-gradient product requires a score-gradient path"
    raise MaterializationError(message)


def _compiled_streaming_gradient_rows(
    execution: StandardExecution,
) -> Iterator[torch.Tensor] | None:
    if execution.compiled_score_matrix is None:
        return None

    boundary = execution.candidate.settings.get("compile.boundary")

    if (execution.operator.kind, boundary) not in {
        ("fisher_vp", "fisher_score_grad"),
        ("sampled_fisher_vp", "sampled_fisher_score_grad"),
        ("empirical_fisher_vp", "empirical_fisher_example_grad"),
        ("per_example_gradient", "per_example_gradient"),
    }:
        return None

    score_gradients = execution.compiled_score_matrix()

    if score_gradients.ndim != MATRIX_DIMS:
        message = "compiled score-gradient rows must be a matrix"
        raise MaterializationError(message)

    _require_finite_tensor(score_gradients, "compiled score-gradient rows")

    return (row.reshape(-1) for row in score_gradients)


def _streaming_gradient_rows_loop(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def tensor_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        output = _call_function_objective(execution, function, active_params)

        if not isinstance(output, torch.Tensor):
            message = "streaming gradient loop requires tensor objective output"
            raise MaterializationError(message)

        return output.reshape(-1)

    terms = tensor_function(*active_leaves)
    _require_nonempty_per_example_terms(terms, "streaming gradient loop")

    for index in range(terms.numel()):
        if not terms[index].requires_grad:
            gradients = tuple(torch.zeros_like(leaf) for leaf in active_leaves)
        else:
            gradient_result = torch.autograd.grad(
                terms[index],
                active_leaves,
                retain_graph=index < terms.numel() - 1,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient.detach()
                for leaf, gradient in zip(
                    active_leaves,
                    gradient_result,
                    strict=True,
                )
            )

        yield _flat_gradient_row(gradients)


def _streaming_gradient_rows_torch_func(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)

    for index in range(terms.numel()):

        def single_loss(
            active: ParameterTree,
            term_index: int = index,
        ) -> torch.Tensor:
            active_terms = _per_example_terms(function, active, execution)

            return active_terms[term_index]

        gradients = torch.func.grad(single_loss)(active_params)
        yield torch.cat(
            tuple(gradients[name].reshape(-1) for name, _ in parameter_items)
        )


def _streaming_gradient_rows_backward(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)

    for index in range(terms.numel()):
        for param in active_params.values():
            param.grad = None

        terms[index].backward(retain_graph=index < terms.numel() - 1)
        gradients = tuple(
            torch.zeros_like(param) if param.grad is None else param.grad.detach()
            for param in active_params.values()
        )

        yield _flat_gradient_row(gradients)


def _streaming_gradient_rows_vmap(
    execution: StandardExecution,
) -> Iterator[torch.Tensor]:
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name].reshape(row_count, -1)
        pieces.append(gradient)

    for index in range(row_count):
        yield torch.cat(tuple(piece[index] for piece in pieces))


def _accumulate_streaming_gradient_row_batch(
    execution: StandardExecution,
    result: torch.Tensor,
    row: torch.Tensor,
    vector_batch: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return row * _dot_runtime(execution.candidate.settings, row, flat_vector)

    return result + _torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)


def _accumulate_streaming_gradient_row(
    execution: StandardExecution,
    result: torch.Tensor,
    row: torch.Tensor,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    return result + row * _dot_runtime(execution.candidate.settings, row, vector_tensor)


def _flat_gradient_row(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))


def _parameter_order_vector(execution: StandardExecution) -> torch.Tensor:
    if execution.flat_parameter_vector is not None:
        return execution.flat_parameter_vector

    return _build_parameter_order_vector(execution)


def _build_parameter_order_vector(execution: StandardExecution) -> torch.Tensor:
    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    vector_tensor = torch.cat(tuple(leaf.reshape(-1) for leaf in vector_leaves))
    _require_finite_tensor(vector_tensor, "streaming Fisher vector")

    return vector_tensor


def _score_matrix_product_batch_vmap(
    execution: StandardExecution,
    score_gradients: torch.Tensor,
    normalization: float,
    label: str,
) -> torch.Tensor:
    vector_batch = _flat_vector_batch(execution)
    _require_score_matrix_product_inputs(
        score_gradients,
        vector_batch,
        f"{label} score_gradients",
    )
    chunk_size = _vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _score_matrix_product(
            score_gradients,
            flat_vector,
            normalization,
            execution.candidate.settings,
            execution.parameter_surface,
        )

    result = _torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)
    _require_finite_tensor(result, f"{label} batched result")

    return result


def _blockwise_score_matrix_product_batch_vmap(
    execution: StandardExecution,
    batch_key: str,
    normalization: float,
    label: str,
) -> torch.Tensor:
    blocks = _loss_scaled_score_blocks(
        execution,
        _batch_tensor_blocks(execution.batch, batch_key),
    )
    vector_batch = _flat_vector_batch(execution)
    _require_blockwise_score_matrix_product_inputs(blocks, vector_batch, label)
    chunk_size = _vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _blockwise_score_matrix_product_unchecked(
            blocks,
            flat_vector,
            normalization,
            execution.candidate.settings,
        )

    result = _torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)
    _require_finite_tensor(result, f"{label} blockwise batched result")

    return result


def _blockwise_score_matrix_product_unchecked(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    first_block = blocks[0]
    offset = first_block.shape[1]
    score_dot = _matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width
        score_dot = score_dot + _matmul_runtime(settings, block, vector[offset:stop])
        offset = stop

    pieces = tuple(_matmul_runtime(settings, block.T, score_dot) for block in blocks)

    return torch.cat(pieces) / normalization


def _flat_vector_batch(execution: StandardExecution) -> torch.Tensor:
    if execution.flat_parameter_vector_batch is not None:
        return execution.flat_parameter_vector_batch

    return _build_flat_vector_batch(execution)


def _build_flat_vector_batch(execution: StandardExecution) -> torch.Tensor:
    vector_in_dims = _vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )

    return _flatten_vector_batch(execution.params, execution.vector, vector_in_dims)


def _require_score_matrix_product_inputs(
    score_gradients: torch.Tensor,
    vector_batch: torch.Tensor,
    label: str,
) -> None:
    if score_gradients.ndim != MATRIX_DIMS:
        message = f"{label} must be a two-dimensional tensor"
        raise MaterializationError(message)

    if vector_batch.ndim != MATRIX_DIMS:
        message = "vectorized Fisher vectors must flatten to a matrix"
        raise MaterializationError(message)

    if score_gradients.shape[0] == 0:
        message = f"{label} must have at least one row"
        raise MaterializationError(message)

    if score_gradients.shape[1] != vector_batch.shape[1]:
        message = f"{label} column count must match vector width"
        raise MaterializationError(message)

    _require_finite_tensor(score_gradients, label)
    _require_finite_tensor(vector_batch, "vectorized Fisher vectors")


def _require_blockwise_score_matrix_product_inputs(
    blocks: tuple[torch.Tensor, ...],
    vector_batch: torch.Tensor,
    label: str,
) -> None:
    if vector_batch.ndim != MATRIX_DIMS:
        message = "vectorized Fisher vectors must flatten to a matrix"
        raise MaterializationError(message)

    if blocks[0].shape[0] == 0:
        message = f"{label} blocks must have at least one row"
        raise MaterializationError(message)

    width = sum(block.shape[1] for block in blocks)

    if width != vector_batch.shape[1]:
        message = f"{label} block columns must match vector width"
        raise MaterializationError(message)

    _require_finite_tensor(vector_batch, "vectorized Fisher vectors")


def _uses_manual_per_example_schedule(execution: StandardExecution) -> bool:
    if execution.candidate.settings.get("schedule.per_example") != "manual_batch":
        return False

    return execution.path in {
        *FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
        *SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
        *EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
    }


def _per_example_gradient_matrix_manual_batches(
    execution: StandardExecution,
) -> torch.Tensor:
    batch, batch_in_dims = _per_example_batch_in_dims(
        execution.batch,
        "per-example manual batching",
    )
    example_count = _per_example_batch_size(
        batch,
        batch_in_dims,
        "per-example manual batching",
    )
    batch_size = _per_example_manual_batch_size(execution)
    rows = []

    for start in range(0, example_count, batch_size):
        stop = min(start + batch_size, example_count)
        subbatch = _per_example_batch_slice(batch, batch_in_dims, start, stop)
        subexecution = dataclasses.replace(execution, batch=subbatch)
        rows.append(_per_example_gradient_matrix_without_manual_batch(subexecution))

    return torch.cat(tuple(rows), dim=0)


def _per_example_gradient_matrix_blockwise(
    execution: StandardExecution,
) -> torch.Tensor:
    batch, batch_in_dims = _per_example_batch_in_dims(
        execution.batch,
        "per-example gradient blockwise stacking",
    )
    example_count = _per_example_batch_size(
        batch,
        batch_in_dims,
        "per-example gradient blockwise stacking",
    )
    block_size = _per_example_block_size(execution)
    rows = []

    for start in range(0, example_count, block_size):
        stop = min(start + block_size, example_count)
        subbatch = _per_example_batch_slice(batch, batch_in_dims, start, stop)
        subexecution = dataclasses.replace(execution, batch=subbatch)
        rows.append(_per_example_gradient_matrix_without_manual_batch(subexecution))

    return torch.cat(tuple(rows), dim=0)


def _per_example_gradient_matrix_without_manual_batch(
    execution: StandardExecution,
) -> torch.Tensor:
    return _per_example_gradient_matrix_from_builders(
        execution,
        PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS,
        (
            "schedule.per_example=manual_batch is incompatible with path: "
            f"{execution.path}"
        ),
    )


def _per_example_manual_batch_size(execution: StandardExecution) -> int:
    if execution.operator.kind in {"fisher_vp", "sampled_fisher_vp"}:
        key = "batch.fisher_sample_batch_size"
    elif execution.operator.kind == "empirical_fisher_vp":
        key = "batch.empirical_example_batch_size"
    else:
        message = "per-example manual batching requires a Fisher-family operator"
        raise MaterializationError(message)

    value = execution.candidate.settings.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _per_example_block_size(execution: StandardExecution) -> int:
    key = "batch.per_example_block_size"
    value = execution.candidate.settings.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _per_example_gradient_matrix(execution: StandardExecution) -> torch.Tensor:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def tensor_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        output = _call_function_objective(execution, function, active_params)

        if not isinstance(output, torch.Tensor):
            message = "per-example gradient loop requires tensor objective output"
            raise MaterializationError(message)

        return output.reshape(-1)

    terms = tensor_function(*active_leaves)

    if terms.numel() == 0:
        message = "per-example gradient loop requires at least one objective term"
        raise MaterializationError(message)

    gradient_rows = []

    for index in range(terms.numel()):
        if not terms[index].requires_grad:
            gradients = tuple(torch.zeros_like(leaf) for leaf in active_leaves)
        else:
            gradient_result = torch.autograd.grad(
                terms[index],
                active_leaves,
                retain_graph=index < terms.numel() - 1,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient.detach()
                for leaf, gradient in zip(
                    active_leaves,
                    gradient_result,
                    strict=True,
                )
            )

        gradient_rows.append(
            torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
        )

    return torch.stack(gradient_rows)


def _loss_scaled_score_matrix(
    execution: StandardExecution,
    matrix: torch.Tensor,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is None:
        return matrix

    return matrix * scale


def _loss_scaled_score_blocks(
    execution: StandardExecution,
    blocks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    scale = _loss_scale(execution.candidate.settings)

    if scale is None:
        return blocks

    return tuple(block * scale for block in blocks)


def _per_example_gradient_matrix_torch_func(
    execution: StandardExecution,
) -> torch.Tensor:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)
    gradient_rows = []

    for index in range(terms.numel()):

        def single_loss(
            active: ParameterTree,
            term_index: int = index,
        ) -> torch.Tensor:
            active_terms = _per_example_terms(function, active, execution)

            return active_terms[term_index]

        gradients = torch.func.grad(single_loss)(active_params)
        pieces = tuple(gradients[name].reshape(-1) for name, _ in parameter_items)
        gradient_rows.append(torch.cat(pieces))

    return torch.stack(gradient_rows)


def _per_example_gradient_matrix_backward(execution: StandardExecution) -> torch.Tensor:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)
    gradient_rows = []

    for index in range(terms.numel()):
        for param in active_params.values():
            param.grad = None

        terms[index].backward(retain_graph=index < terms.numel() - 1)
        pieces = tuple(
            torch.zeros_like(param).reshape(-1)
            if param.grad is None
            else param.grad.detach().reshape(-1)
            for param in active_params.values()
        )
        gradient_rows.append(torch.cat(pieces))

    return torch.stack(gradient_rows)


def _per_example_terms(
    function: FunctionObjective,
    active_params: ParameterTree,
    execution: StandardExecution,
) -> torch.Tensor:
    output = _call_function_objective(execution, function, active_params)

    if not isinstance(output, torch.Tensor):
        message = "per-example gradient path requires tensor objective output"
        raise MaterializationError(message)

    terms = output.reshape(-1)

    _require_nonempty_per_example_terms(terms, "per-example gradient path")

    return terms


def _require_nonempty_per_example_terms(terms: torch.Tensor, label: str) -> None:
    if terms.numel() == 0:
        message = f"{label} requires at least one objective term"
        raise MaterializationError(message)


def _per_example_gradient_matrix_vmap(execution: StandardExecution) -> torch.Tensor:
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name]
        pieces.append(gradient.reshape(row_count, -1))

    return torch.cat(tuple(pieces), dim=1)


def _per_example_gradient_matrix_from_builders(
    execution: StandardExecution,
    builders: Mapping[str, Callable[[StandardExecution], torch.Tensor]],
    message: str,
) -> torch.Tensor:
    builder = builders.get(execution.path)

    if builder is None:
        raise MaterializationError(message)

    return builder(execution)


FISHER_SCORE_GRADIENT_BUILDERS = {
    FISHER_SCORE_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH: _per_example_gradient_matrix_torch_func,
    FISHER_SCORE_GRADIENT_VMAP_PATH: _per_example_gradient_matrix_vmap,
    FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
}
SAMPLED_FISHER_SCORE_GRADIENT_BUILDERS = {
    SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH: (
        _per_example_gradient_matrix_torch_func
    ),
    SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH: _per_example_gradient_matrix_vmap,
    SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
}
EMPIRICAL_FISHER_GRADIENT_BUILDERS = {
    EMPIRICAL_FISHER_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH: _per_example_gradient_matrix_torch_func,
    EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
    EMPIRICAL_FISHER_GRADIENT_VMAP_PATH: _per_example_gradient_matrix_vmap,
}
PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS = {
    FISHER_SCORE_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    EMPIRICAL_FISHER_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    PER_EXAMPLE_GRADIENT_LOOP_PATH: _per_example_gradient_matrix,
    FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH: _per_example_gradient_matrix_torch_func,
    SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH: (
        _per_example_gradient_matrix_torch_func
    ),
    EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH: _per_example_gradient_matrix_torch_func,
    PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH: _per_example_gradient_matrix_torch_func,
    FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
    SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
    EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH: _per_example_gradient_matrix_backward,
    PER_EXAMPLE_GRADIENT_BACKWARD_PATH: _per_example_gradient_matrix_backward,
    PER_EXAMPLE_GRADIENT_VMAP_PATH: _per_example_gradient_matrix_vmap,
}


def _per_example_gradient_tree_vmap(
    execution: StandardExecution,
) -> tuple[ParameterTree, tuple[tuple[str, torch.Tensor], ...], int]:
    try:
        admit_torch_func(execution.candidate.settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error

    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    batched_batch, batch_in_dims = _per_example_vmap_batch(execution.batch)
    chunk_size = _per_example_vmap_chunk_size(execution)

    def single_loss(
        active_params: ParameterTree,
        single_tensor_batch: Batch,
    ) -> torch.Tensor:
        output = _call_function_objective(
            execution,
            function,
            active_params,
            single_tensor_batch,
        )

        if not isinstance(output, torch.Tensor):
            message = "per-example vmap requires tensor objective output"
            raise MaterializationError(message)

        terms = output.reshape(-1)

        if terms.numel() != 1:
            message = "per-example vmap objective must return one scalar per example"
            raise MaterializationError(message)

        return terms[0]

    gradients = _torch_func_vmap(
        _torch_func_grad(single_loss),
        in_dims=(None, batch_in_dims),
        randomness=str(execution.candidate.settings["vectorization.randomness"]),
        chunk_size=chunk_size,
    )(active_params, batched_batch)
    row_count = _vmap_batch_size(batched_batch, batch_in_dims)

    return gradients, parameter_items, row_count


def _per_example_vmap_batch(
    batch: Batch,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    return _per_example_batch_in_dims(batch, "per-example vmap")


def _per_example_batch_in_dims(
    batch: Batch,
    label: str,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    result = {}
    in_dims = {}
    expected_size = None

    for key, value in batch.items():
        result[key] = value

        if not isinstance(value, torch.Tensor):
            in_dims[key] = None
            continue

        if value.ndim == 0:
            message = f"{label} tensor batch field is scalar: {key}"
            raise MaterializationError(message)

        leading_size = value.shape[0]

        if expected_size is None:
            expected_size = leading_size
        elif leading_size != expected_size:
            message = f"{label} batch leading dimensions differ"
            raise MaterializationError(message)

        in_dims[key] = 0

    if expected_size is None or expected_size == 0:
        message = f"{label} requires a nonempty mapped batch"
        raise MaterializationError(message)

    return result, in_dims


def _per_example_batch_size(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
    label: str,
) -> int:
    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            if dim < 0:
                dim += value.ndim

            return value.shape[dim]

    message = f"{label} requires a nonempty mapped batch"
    raise MaterializationError(message)


def _per_example_batch_slice(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
    start: int,
    stop: int,
) -> dict[str, Any]:
    result = {}

    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            result[key] = value.narrow(dim, start, stop - start)
        else:
            result[key] = value

    return result


def _per_example_vmap_chunk_size(execution: StandardExecution) -> int | None:
    if execution.operator.kind in {"fisher_vp", "sampled_fisher_vp"}:
        key = "batch.fisher_sample_batch_size"
    elif execution.operator.kind == "empirical_fisher_vp":
        key = "batch.empirical_example_batch_size"
    elif execution.operator.kind == "per_example_gradient":
        return None
    else:
        message = "per-example vmap chunk size requires a Fisher-family operator"
        raise MaterializationError(message)

    chunk_size = execution.candidate.settings.get(key)

    if chunk_size is None:
        return None

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return chunk_size


def _vmap_batch_size(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
) -> int:
    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            if dim < 0:
                dim += value.ndim

            return value.shape[dim]

    message = "per-example vmap requires a nonempty batch"
    raise MaterializationError(message)


def _vmap_chunk_size(settings: Mapping[str, Any]) -> int | None:
    chunk_size = settings.get("vectorization.vmap_chunk_size")

    if chunk_size is None:
        message = "vectorization.mode=vmap requires vectorization.vmap_chunk_size"
        raise MaterializationError(message)

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        message = "vectorization.vmap_chunk_size must be a positive integer"
        raise MaterializationError(message)

    return chunk_size


def _manual_vector_batch_size(settings: Mapping[str, Any]) -> int:
    batch_size = settings.get("vectorization.batch_size")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        message = (
            "vectorization.mode=manual_batch requires positive vectorization.batch_size"
        )
        raise MaterializationError(message)

    return batch_size


def _vector_tree_in_dims(
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> Any:
    raw_in_dims = settings.get("vectorization.in_dims")

    if raw_in_dims is None:
        message = "vectorized vector inputs require vectorization.in_dims"
        raise MaterializationError(message)

    return _validate_vector_tree_in_dims(vector, raw_in_dims)


def _validate_vector_tree_in_dims(vector: TensorTree, raw_in_dims: Any) -> Any:
    if isinstance(vector, torch.Tensor):
        return _validate_vector_tensor_in_dim(vector, raw_in_dims)

    if _is_tensor_tree_dict(vector):
        return _validate_vector_dict_in_dims(vector, raw_in_dims)

    if _is_tensor_tree_tuple(vector):
        return _validate_vector_tuple_in_dims(vector, raw_in_dims)

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise MaterializationError(message)


def _validate_vector_tensor_in_dim(
    vector: torch.Tensor,
    raw_in_dim: Any,
) -> int | None:
    if raw_in_dim is None:
        return None

    if not isinstance(raw_in_dim, int) or isinstance(raw_in_dim, bool):
        message = "vectorization.in_dims values must be integers or None"
        raise MaterializationError(message)

    dim = raw_in_dim

    if dim < 0:
        dim += vector.ndim

    if dim < 0 or dim >= vector.ndim:
        message = "vectorization.in_dims axis is out of range"
        raise MaterializationError(message)

    if vector.shape[dim] == 0:
        message = "vectorized vector inputs require a nonempty mapped dimension"
        raise MaterializationError(message)

    return raw_in_dim


def _validate_vector_dict_in_dims(
    vector: dict[str, TensorTree],
    raw_in_dims: Any,
) -> dict[str, Any]:
    if not isinstance(raw_in_dims, Mapping):
        message = "vectorization.in_dims must match the vector tree"
        raise MaterializationError(message)

    if set(raw_in_dims) != set(vector):
        message = "vectorization.in_dims must cover every vector key"
        raise MaterializationError(message)

    return {
        key: _validate_vector_tree_in_dims(vector[key], raw_in_dims[key])
        for key in vector
    }


def _validate_vector_tuple_in_dims(
    vector: tuple[TensorTree, ...],
    raw_in_dims: Any,
) -> tuple[Any, ...]:
    if not isinstance(raw_in_dims, tuple):
        message = "vectorization.in_dims must match the vector tree"
        raise MaterializationError(message)

    if len(raw_in_dims) != len(vector):
        message = "vectorization.in_dims must cover every vector element"
        raise MaterializationError(message)

    result = []

    for value, in_dim in zip(vector, raw_in_dims, strict=True):
        result.append(_validate_vector_tree_in_dims(value, in_dim))

    return tuple(result)


def _vector_tree_batch_size(vector: TensorTree, in_dims: Any) -> int:
    sizes = []
    _collect_vector_tree_batch_sizes(vector, in_dims, sizes)

    if not sizes:
        message = "vectorized vector inputs require at least one mapped leaf"
        raise MaterializationError(message)

    first_size = sizes[0]

    for size in sizes[1:]:
        if size != first_size:
            message = "vectorized vector mapped dimensions differ"
            raise MaterializationError(message)

    return first_size


def _collect_vector_tree_batch_sizes(
    vector: TensorTree,
    in_dims: Any,
    sizes: list[int],
) -> None:
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return

        dim = _normalized_vector_dim(vector, in_dims)
        sizes.append(vector.shape[dim])
        return

    if _is_tensor_tree_dict(vector):
        for key in vector:
            _collect_vector_tree_batch_sizes(vector[key], in_dims[key], sizes)

        return

    if _is_tensor_tree_tuple(vector):
        for value, in_dim in zip(vector, in_dims, strict=True):
            _collect_vector_tree_batch_sizes(value, in_dim, sizes)

        return

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise MaterializationError(message)


def _vector_tree_select(vector: TensorTree, in_dims: Any, index: int) -> TensorTree:
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return vector

        dim = _normalized_vector_dim(vector, in_dims)

        return vector.select(dim, index)

    if _is_tensor_tree_dict(vector):
        return {
            key: _vector_tree_select(vector[key], in_dims[key], index) for key in vector
        }

    if _is_tensor_tree_tuple(vector):
        return tuple(
            _vector_tree_select(value, in_dim, index)
            for value, in_dim in zip(vector, in_dims, strict=True)
        )

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise MaterializationError(message)


def _vector_tree_slice(
    vector: TensorTree,
    in_dims: Any,
    start: int,
    stop: int,
) -> TensorTree:
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return vector

        dim = _normalized_vector_dim(vector, in_dims)

        return vector.narrow(dim, start, stop - start)

    if _is_tensor_tree_dict(vector):
        return {
            key: _vector_tree_slice(vector[key], in_dims[key], start, stop)
            for key in vector
        }

    if _is_tensor_tree_tuple(vector):
        return tuple(
            _vector_tree_slice(value, in_dim, start, stop)
            for value, in_dim in zip(vector, in_dims, strict=True)
        )

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise MaterializationError(message)


def _normalized_vector_dim(vector: torch.Tensor, in_dim: int) -> int:
    dim = in_dim

    if dim < 0:
        dim += vector.ndim

    if dim < 0 or dim >= vector.ndim:
        message = "vectorization.in_dims axis is out of range"
        raise MaterializationError(message)

    return dim


def _stack_tensor_trees(outputs: Sequence[TensorTree], dim: int) -> TensorTree:
    if not outputs:
        message = "cannot stack an empty tensor-tree sequence"
        raise MaterializationError(message)

    first = outputs[0]

    if isinstance(first, torch.Tensor):
        leaves = []

        for output in outputs:
            if not isinstance(output, torch.Tensor):
                message = "tensor tree structures differ"
                raise MaterializationError(message)

            leaves.append(output)

        return torch.stack(tuple(leaves), dim=dim)

    if _is_tensor_tree_dict(first):
        return _stack_tensor_tree_dicts(outputs, first, dim)

    if _is_tensor_tree_tuple(first):
        return _stack_tensor_tree_tuples(outputs, first, dim)

    message = f"unsupported tensor tree node: {type(first).__name__}"
    raise MaterializationError(message)


def _cat_tensor_trees(outputs: Sequence[TensorTree], dim: int) -> TensorTree:
    if not outputs:
        message = "cannot concatenate an empty tensor-tree sequence"
        raise MaterializationError(message)

    first = outputs[0]

    if isinstance(first, torch.Tensor):
        leaves = []

        for output in outputs:
            if not isinstance(output, torch.Tensor):
                message = "tensor tree structures differ"
                raise MaterializationError(message)

            leaves.append(output)

        return torch.cat(tuple(leaves), dim=dim)

    if _is_tensor_tree_dict(first):
        dict_outputs = _require_tensor_tree_dict_outputs(outputs, first)

        return {
            key: _cat_tensor_trees(tuple(output[key] for output in dict_outputs), dim)
            for key in first
        }

    if _is_tensor_tree_tuple(first):
        tuple_outputs = _require_tensor_tree_tuple_outputs(outputs, first)

        return tuple(
            _cat_tensor_trees(tuple(output[index] for output in tuple_outputs), dim)
            for index in range(len(first))
        )

    message = f"unsupported tensor tree node: {type(first).__name__}"
    raise MaterializationError(message)


def _stack_tensor_tree_dicts(
    outputs: Sequence[TensorTree],
    first: dict[str, TensorTree],
    dim: int,
) -> TensorTree:
    dict_outputs = _require_tensor_tree_dict_outputs(outputs, first)
    result = {}

    for key in first:
        child_outputs = tuple(output[key] for output in dict_outputs)
        result[key] = _stack_tensor_trees(tuple(child_outputs), dim)

    return result


def _require_tensor_tree_dict_outputs(
    outputs: Sequence[TensorTree],
    first: dict[str, TensorTree],
) -> tuple[dict[str, TensorTree], ...]:
    result = []

    for output in outputs:
        if not _is_tensor_tree_dict(output) or set(output) != set(first):
            message = "tensor tree mapping keys differ"
            raise MaterializationError(message)

        result.append(output)

    return tuple(result)


def _stack_tensor_tree_tuples(
    outputs: Sequence[TensorTree],
    first: tuple[TensorTree, ...],
    dim: int,
) -> TensorTree:
    tuple_outputs = _require_tensor_tree_tuple_outputs(outputs, first)
    result = []

    for index in range(len(first)):
        child_outputs = tuple(output[index] for output in tuple_outputs)
        result.append(_stack_tensor_trees(tuple(child_outputs), dim))

    return tuple(result)


def _require_tensor_tree_tuple_outputs(
    outputs: Sequence[TensorTree],
    first: tuple[TensorTree, ...],
) -> tuple[tuple[TensorTree, ...], ...]:
    result = []

    for output in outputs:
        if not _is_tensor_tree_tuple(output) or len(output) != len(first):
            message = "tensor tree sequence lengths differ"
            raise MaterializationError(message)

        result.append(output)

    return tuple(result)


def _torch_func_grad(function: Callable[..., torch.Tensor]) -> Callable[..., Any]:
    return torch.func.grad(function)


def _torch_func_vmap(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return torch.func.vmap(function, **kwargs)


def _run_metric(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            METRIC_DENSE_PATH,
            METRIC_FACTORIZED_PATH,
            METRIC_BLOCKWISE_PATH,
            METRIC_STREAMING_PATH,
        ),
    )
    _require_metric_accumulation_settings(
        execution.path,
        execution.candidate.settings,
    )
    if execution.compiled_inner is None:
        result = _metric_multiply_by_path(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.path,
            execution.candidate.settings,
        )
    else:
        result = execution.compiled_inner()

    _require_finite_tree(result, "metric result")

    return result


def _run_sqrt_metric(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            SQRT_METRIC_CLOSED_FORM_PATH,
            SQRT_METRIC_CHOLESKY_PATH,
            SQRT_METRIC_EIGENBASIS_PATH,
            SQRT_METRIC_LANCZOS_PATH,
        ),
    )
    if execution.compiled_inner is None:
        result = _metric_square_root_apply(
            execution,
            inverse=execution.operator.kind == "inverse_sqrt_metric",
            adjoint=False,
        )
    else:
        result = execution.compiled_inner()

    _require_finite_tree(result, "metric square-root result")

    return result


def _run_metric_inner(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            METRIC_INNER_MULTIPLY_REDUCE_PATH,
            METRIC_INNER_FACTORED_GRAM_PATH,
            METRIC_INNER_SQRT_REDUCE_PATH,
        ),
    )
    _require_metric_inner_norm_path(execution, "metric_inner.reduction_path")
    if execution.compiled_inner is None:
        left, right = _metric_inner_vectors(execution.vector)
        result = _metric_inner_by_path(execution, left, right)
    else:
        result = _metric_inner_compiled_tensor(
            execution.compiled_inner(),
            "metric inner result",
        )

    _require_finite_tensor(result, "metric inner result")

    return result


def _run_inverse_metric_inner(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
            INVERSE_METRIC_INNER_FACTORED_GRAM_PATH,
            INVERSE_METRIC_INNER_SQRT_REDUCE_PATH,
        ),
    )
    _require_metric_inner_norm_path(execution, "inverse_metric_inner.reduction_path")
    if execution.compiled_inner is None:
        left, right = _metric_inner_vectors(execution.vector)
        result = _inverse_metric_inner_by_path(execution, left, right)
    else:
        result = _metric_inner_compiled_tensor(
            execution.compiled_inner(),
            "inverse metric inner result",
        )

    _require_finite_tensor(result, "inverse metric inner result")

    return result


def _metric_inner_compiled_tensor(result: TensorTree, name: str) -> torch.Tensor:
    if not isinstance(result, torch.Tensor):
        message = f"{name} must be a tensor"
        raise MaterializationError(message)

    return result


def _require_metric_inner_norm_path(
    execution: StandardExecution,
    reduction_key: str,
) -> None:
    if execution.operator.semantics.get("as_norm") is not True:
        return

    if execution.candidate.settings.get(reduction_key) == "sqrt_apply_reduce":
        return

    message = f"{reduction_key}=sqrt_apply_reduce is required when as_norm=True"
    raise MaterializationError(message)


def _metric_inner_vectors(vector: TensorTree) -> tuple[TensorTree, TensorTree]:
    if not isinstance(vector, tuple) or len(vector) != METRIC_INNER_VECTOR_COUNT:
        message = "metric inner vector input must be a (left, right) tuple"
        raise MaterializationError(message)

    left, right = vector

    return left, right


def _metric_inner_by_path(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    if execution.path == METRIC_INNER_MULTIPLY_REDUCE_PATH:
        return _metric_inner_multiply_then_reduce(execution, left, right)

    if execution.path == METRIC_INNER_FACTORED_GRAM_PATH:
        return _metric_inner_factored_gram(execution, left, right)

    if execution.path == METRIC_INNER_SQRT_REDUCE_PATH:
        return _metric_inner_sqrt_apply_reduce(
            execution,
            left,
            right,
            inverse=False,
        )

    message = f"metric inner path is not lowered: {execution.path}"
    raise MaterializationError(message)


def _inverse_metric_inner_by_path(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    if execution.path == INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        return _inverse_metric_inner_solve_then_reduce(execution, left, right)

    if execution.path == INVERSE_METRIC_INNER_FACTORED_GRAM_PATH:
        return _inverse_metric_inner_factored_gram(execution, left, right)

    if execution.path == INVERSE_METRIC_INNER_SQRT_REDUCE_PATH:
        return _metric_inner_sqrt_apply_reduce(
            execution,
            left,
            right,
            inverse=True,
        )

    message = f"inverse metric inner path is not lowered: {execution.path}"
    raise MaterializationError(message)


def _metric_inner_multiply_then_reduce(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)

    return _metric_inner_multiply_reduce_by_metric_path(
        execution,
        left,
        right,
        metric_path,
        "metric_inner.multi_rhs",
    )


def _inverse_metric_inner_solve_then_reduce(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    inverse_path = _inverse_metric_runtime_path_from_settings(
        execution.candidate.settings
    )

    return _metric_inner_inverse_reduce_by_path(
        execution,
        left,
        right,
        inverse_path,
        "inverse_metric_inner.multi_rhs",
    )


def _metric_inner_factored_gram(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_factorized_multiply_reduce(execution, left, right)


def _inverse_metric_inner_factored_gram(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_inverse_reduce_by_path(
        execution,
        left,
        right,
        INVERSE_METRIC_FACTORIZED_PATH,
        "inverse_metric_inner.multi_rhs",
    )


def _metric_inner_factorized_multiply_reduce(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_multiply_reduce_by_metric_path(
        execution,
        left,
        right,
        METRIC_FACTORIZED_PATH,
        "metric_inner.multi_rhs",
    )


def _metric_inner_multiply_reduce_by_metric_path(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
    metric_path: str,
    multi_rhs_key: str,
) -> torch.Tensor:
    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_inner_flat_block(execution, left_block, 0)

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        right_matrix = _metric_inner_flat_block(execution, right_block, 1)

        return _metric_apply_flat_batch(
            execution.operator,
            execution.batch,
            execution.params,
            right_matrix,
            0.0,
            metric_path,
            execution.candidate.settings,
        )

    def right_vector_product(right_vector: TensorTree) -> TensorTree:
        return _metric_multiply_by_path(
            execution.operator,
            execution.batch,
            right_vector,
            metric_path,
            execution.candidate.settings,
        )

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=right_vector_product,
        left_vector_product=None,
    )


def _metric_inner_inverse_reduce_by_path(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
    inverse_path: str,
    multi_rhs_key: str,
) -> torch.Tensor:
    solve_execution = dataclasses.replace(
        execution,
        path=inverse_path,
        vector=right,
    )

    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_inner_flat_block(execution, left_block, 0)

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        right_execution = _metric_inner_side_execution(
            solve_execution,
            right_block,
            1,
            path=inverse_path,
        )
        product = _run_inverse_metric_rhs_batch(right_execution)

        return _metric_inner_flat_leading_block(execution, product)

    def right_vector_product(right_vector: TensorTree) -> TensorTree:
        right_execution = dataclasses.replace(
            solve_execution,
            vector=right_vector,
        )

        return _inverse_metric_solve_by_path(right_execution)

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=right_vector_product,
        left_vector_product=None,
    )


def _metric_inner_sqrt_apply_reduce(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
    *,
    inverse: bool,
) -> torch.Tensor:
    multi_rhs_key = (
        "inverse_metric_inner.multi_rhs" if inverse else "metric_inner.multi_rhs"
    )
    sqrt_path = _sqrt_metric_runtime_path_from_settings(execution.candidate.settings)

    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_square_root_apply_flat_batch(
            execution,
            left_block,
            0,
            sqrt_path,
            inverse=inverse,
        )

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        return _metric_square_root_apply_flat_batch(
            execution,
            right_block,
            1,
            sqrt_path,
            inverse=inverse,
        )

    def factor_vector(vector: TensorTree) -> TensorTree:
        return _metric_square_root_apply_vector(
            execution,
            vector,
            sqrt_path,
            inverse=inverse,
            adjoint=True,
        )

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=factor_vector,
        left_vector_product=factor_vector,
    )


def _metric_inner_reduce_by_product_builders(
    execution: StandardExecution,
    left: TensorTree,
    right: TensorTree,
    multi_rhs_key: str,
    *,
    left_matrix_builder: Callable[[TensorTree], torch.Tensor],
    right_matrix_builder: Callable[[TensorTree], torch.Tensor],
    right_vector_product: Callable[[TensorTree], TensorTree],
    left_vector_product: Callable[[TensorTree], TensorTree] | None,
) -> torch.Tensor:
    block_mode = _metric_inner_block_mode(execution, multi_rhs_key)

    if block_mode is not None:
        left_matrix = left_matrix_builder(left)

        if block_mode == "manual_batch":
            return _metric_inner_manual_block_reduce(
                execution,
                left_matrix,
                right,
                right_matrix_builder,
            )

        if block_mode == "vmap":
            return _metric_inner_vmap_block_reduce(
                execution,
                left_matrix,
                right,
                1,
                right_vector_product,
            )

        right_matrix = right_matrix_builder(right)

        return _metric_inner_reduce_matrices(execution, left_matrix, right_matrix)

    if left_vector_product is None:
        right_product = right_vector_product(right)

        return _tree_dot_runtime(execution.candidate.settings, left, right_product)

    left_product = left_vector_product(left)
    right_product = right_vector_product(right)

    return _tree_dot_runtime(execution.candidate.settings, left_product, right_product)


def _metric_inner_block_mode(
    execution: StandardExecution,
    key: str,
) -> str | None:
    if key not in execution.candidate.settings:
        message = f"{key} is required"
        raise MaterializationError(message)

    value = execution.candidate.settings[key]

    if value == "single_column":
        if "vectorization.mode" in execution.candidate.settings:
            message = f"vectorization.mode requires {key}=block"
            raise MaterializationError(message)

        return None

    if value != "block":
        message = f"{key} is unsupported: {value}"
        raise MaterializationError(message)

    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "single_loop":
        return mode

    if mode == "manual_batch":
        _manual_vector_batch_size(execution.candidate.settings)

        return mode

    if mode == "vmap":
        _vmap_chunk_size(execution.candidate.settings)

        return mode

    message = (
        f"{key}=block requires vectorization.mode=single_loop, manual_batch, or vmap"
    )
    raise MaterializationError(message)


def _metric_inner_vmap_block_reduce(
    execution: StandardExecution,
    left_matrix: torch.Tensor,
    right: TensorTree,
    side: int,
    right_product: Callable[[TensorTree], TensorTree],
) -> torch.Tensor:
    right_in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        right,
        side,
    )
    chunk_size = _vmap_chunk_size(execution.candidate.settings)

    def flat_right_product(right_vector: TensorTree) -> torch.Tensor:
        return _flatten_vector(
            _call_with_deferred_finite_checks(right_product, right_vector)
        )

    product_matrix = _torch_func_vmap(
        flat_right_product,
        in_dims=(right_in_dims,),
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(right)
    _require_finite_tensor(product_matrix, "metric inner vmap block result")

    return _metric_inner_reduce_matrices(execution, left_matrix, product_matrix)


def _metric_inner_manual_block_reduce(
    execution: StandardExecution,
    left_matrix: torch.Tensor,
    right: TensorTree,
    right_matrix_builder: Callable[[TensorTree], torch.Tensor],
) -> torch.Tensor:
    parts = []

    for right_chunk in _metric_inner_manual_right_chunks(execution, right):
        right_matrix = right_matrix_builder(right_chunk)
        parts.append(
            _metric_inner_reduce_matrices(execution, left_matrix, right_matrix)
        )

    result = torch.cat(tuple(parts), dim=1)
    _require_finite_tensor(result, "metric inner manual-batch block result")

    return result


def _metric_inner_manual_right_chunks(
    execution: StandardExecution,
    right: TensorTree,
) -> tuple[TensorTree, ...]:
    right_in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        right,
        1,
    )
    right_count = _vector_tree_batch_size(right, right_in_dims)
    batch_size = _manual_vector_batch_size(execution.candidate.settings)
    chunks = []

    for start in range(0, right_count, batch_size):
        stop = min(start + batch_size, right_count)
        chunks.append(_vector_tree_slice(right, right_in_dims, start, stop))

    return tuple(chunks)


def _metric_inner_side_execution(
    execution: StandardExecution,
    vector: TensorTree,
    side: int,
    *,
    path: str,
) -> StandardExecution:
    settings = _metric_inner_side_settings(execution.candidate.settings, side)
    candidate = dataclasses.replace(execution.candidate, settings=settings)

    return dataclasses.replace(
        execution,
        candidate=candidate,
        path=path,
        vector=vector,
    )


def _metric_inner_side_settings(
    settings: Mapping[str, Any],
    side: int,
) -> Mapping[str, Any]:
    raw_in_dims = settings.get("vectorization.in_dims")

    if not isinstance(raw_in_dims, tuple):
        return settings

    if len(raw_in_dims) != METRIC_INNER_VECTOR_COUNT:
        message = "metric inner vectorization.in_dims must cover left and right"
        raise MaterializationError(message)

    result = dict(settings)
    result["vectorization.in_dims"] = raw_in_dims[side]

    return result


def _metric_inner_flat_leading_block(
    execution: StandardExecution,
    vector: TensorTree,
) -> torch.Tensor:
    return _flatten_vector_batch(
        execution.params,
        vector,
        _leading_vector_batch_in_dims(execution.params),
    )


def _leading_vector_batch_in_dims(template: TensorTree) -> Any:
    if isinstance(template, torch.Tensor):
        return 0

    if _is_tensor_tree_dict(template):
        return {key: _leading_vector_batch_in_dims(template[key]) for key in template}

    if _is_tensor_tree_tuple(template):
        return tuple(_leading_vector_batch_in_dims(value) for value in template)

    message = f"unsupported tensor tree node: {type(template).__name__}"
    raise MaterializationError(message)


def _metric_inner_flat_block(
    execution: StandardExecution,
    vector: TensorTree,
    side: int,
) -> torch.Tensor:
    in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        vector,
        side,
    )

    return _flatten_vector_batch(execution.params, vector, in_dims)


def _metric_inner_vector_in_dims(
    settings: Mapping[str, Any],
    vector: TensorTree,
    side: int,
) -> Any:
    raw_in_dims = settings.get("vectorization.in_dims")

    if isinstance(raw_in_dims, tuple):
        if len(raw_in_dims) != METRIC_INNER_VECTOR_COUNT:
            message = "metric inner vectorization.in_dims must cover left and right"
            raise MaterializationError(message)

        raw_in_dims = raw_in_dims[side]

    return _validate_vector_tree_in_dims(vector, raw_in_dims)


def _metric_inner_reduce_matrices(
    execution: StandardExecution,
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
) -> torch.Tensor:
    if left_matrix.ndim != MATRIX_DIMS or right_matrix.ndim != MATRIX_DIMS:
        message = "metric inner block operands must flatten to matrices"
        raise MaterializationError(message)

    if left_matrix.shape[1] != right_matrix.shape[1]:
        message = "metric inner block widths differ"
        raise MaterializationError(message)

    result = _matmul_runtime(execution.candidate.settings, left_matrix, right_matrix.T)
    _require_finite_tensor(result, "metric inner block result")

    return result


def _sqrt_metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return _runtime_path_from_settings(
        settings,
        "sqrt_metric",
        "sqrt_metric.factor_path",
    )


def _runtime_path_from_settings(
    settings: Mapping[str, Any],
    operator_kind: str,
    setting_key: str,
) -> str:
    value = settings.get(setting_key)

    if not isinstance(value, str):
        message = f"{setting_key} is required"
        raise MaterializationError(message)

    path = SPEC_PATH_TO_RUNTIME[operator_kind].get(value)

    if path is None:
        message = f"{setting_key} value is not lowered: {value}"
        raise MaterializationError(message)

    return path


def _metric_square_root_apply_vector(
    execution: StandardExecution,
    vector: TensorTree,
    path: str,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    sqrt_execution = dataclasses.replace(
        execution,
        path=path,
        vector=vector,
    )

    return _metric_square_root_apply(sqrt_execution, inverse=inverse, adjoint=adjoint)


def _metric_square_root_apply_flat_batch(
    execution: StandardExecution,
    vector: TensorTree,
    side: int,
    path: str,
    *,
    inverse: bool,
) -> torch.Tensor:
    flat_vectors = _metric_inner_flat_block(execution, vector, side)
    rows = []

    for flat_vector in flat_vectors:
        vector_tree = _wrap_flat_vector(execution.params, flat_vector)
        result = _metric_square_root_apply_vector(
            execution,
            vector_tree,
            path,
            inverse=inverse,
            adjoint=True,
        )
        rows.append(_flatten_vector(result))

    matrix = torch.stack(tuple(rows), dim=0)
    _require_finite_tensor(matrix, "metric square-root block result")

    return matrix


def _metric_square_root_apply(
    execution: StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    path = execution.path
    vector = execution.vector

    if path == SQRT_METRIC_CLOSED_FORM_PATH:
        return _closed_form_metric_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    if path == SQRT_METRIC_LANCZOS_PATH:
        _require_metric_representation(execution.operator, ("matrix_free",))
        flat_result = _lanczos_matrix_free_metric_square_root_product(
            execution,
            _flatten_vector(vector),
            inverse=inverse,
        )
        _require_finite_tensor(flat_result, "metric square-root result")

        return _wrap_flat_vector(vector, flat_result)

    matrix = _metric_dense_matrix(execution.operator, execution.batch, vector)

    factor = _metric_square_root_factor_matrix(
        execution,
        matrix,
        inverse=inverse,
        path=path,
    )
    flat_vector = _flatten_vector(vector)
    flat_result = factor.T @ flat_vector if adjoint else factor @ flat_vector

    _require_finite_tensor(flat_result, "metric square-root result")

    return _wrap_flat_vector(vector, flat_result)


def _closed_form_metric_square_root_apply(
    execution: StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    representation = _metric_representation_kind(execution.operator)

    if representation == "ekfac_factors":
        return _ekfac_square_root_apply(execution, inverse=inverse)

    if representation == "kfac_factors":
        return _kfac_square_root_apply(execution, inverse=inverse)

    if representation == "low_rank_factors":
        return _low_rank_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    if representation == "ggn_derived_factors":
        return _ggn_derived_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    _require_metric_representation(execution.operator, ("diagonal_tree",))
    diagonal = _flatten_vector(_metric_diagonal_tree(execution.batch, execution.vector))

    if inverse:
        damping = _inverse_metric_damping(execution.operator)
        factors = torch.rsqrt(diagonal + damping)
    else:
        _require_positive_spectrum(diagonal, "diagonal metric square root")
        factors = torch.sqrt(diagonal)

    flat_result = factors * _flatten_vector(execution.vector)
    _require_finite_tensor(flat_result, "closed-form metric square-root result")

    return _wrap_flat_vector(execution.vector, flat_result)


def _rectangular_square_root_apply(
    execution: StandardExecution,
    *,
    adjoint: bool,
    name: str,
    widths: tuple[int, int],
    width_labels: tuple[str, str],
    products: Sequence[Callable[[torch.Tensor], torch.Tensor]],
) -> TensorTree:
    branch = 0 if adjoint else 1
    role = ("adjoint ", "")[branch]
    flat_vector = _flatten_vector(execution.vector)

    if flat_vector.numel() != widths[branch]:
        message = f"{name} {role}input must match {width_labels[branch]}"
        raise MaterializationError(message)

    flat_result = products[branch](flat_vector)
    _require_finite_tensor(flat_result, f"{name} {role}result")

    if adjoint:
        return flat_result

    return _wrap_flat_vector(execution.params, flat_result)


def _low_rank_square_root_apply(
    execution: StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    basis, diagonal = _low_rank_factors(execution.batch, execution.params)

    if inverse:
        base_diagonal = diagonal + _inverse_metric_damping(execution.operator)
        flat_result = _low_rank_plus_diagonal_inverse_square_root_flat_apply(
            basis,
            base_diagonal,
            _flatten_vector(execution.vector),
            adjoint=adjoint,
        )

        return _wrap_flat_vector(execution.params, flat_result)

    _require_nonnegative_spectrum(diagonal, "low-rank diagonal square root")
    rank = basis.shape[1]
    width = diagonal.numel()
    root_diagonal = torch.sqrt(diagonal)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return basis @ flat_vector[:rank] + root_diagonal * flat_vector[rank:]

    return _rectangular_square_root_apply(
        execution,
        adjoint=adjoint,
        name="low-rank square-root",
        widths=(width, rank + width),
        width_labels=("parameter width", "rank plus parameter width"),
        products=(
            lambda flat_vector: torch.cat((
                basis.T @ flat_vector,
                root_diagonal * flat_vector,
            )),
            product,
        ),
    )


def _low_rank_plus_diagonal_inverse_square_root_flat_apply(
    basis: torch.Tensor,
    base_diagonal: torch.Tensor,
    vector: torch.Tensor,
    *,
    adjoint: bool,
) -> torch.Tensor:
    _require_positive_spectrum(base_diagonal, "low-rank inverse square root base")
    root_base = torch.sqrt(base_diagonal)
    scaled_basis = basis / root_base.unsqueeze(1)

    if adjoint:
        base_scaled = vector / root_base
        result = _identity_plus_low_rank_inverse_square_root_apply(
            scaled_basis,
            base_scaled,
        )
    else:
        reduced = _identity_plus_low_rank_inverse_square_root_apply(
            scaled_basis,
            vector,
        )
        result = reduced / root_base

    _require_finite_tensor(result, "low-rank inverse square-root result")

    return result


def _identity_plus_low_rank_inverse_square_root_apply(
    basis: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    if basis.shape[0] != vector.numel():
        message = "inverse square-root basis width differs from vector"
        raise MaterializationError(message)

    if basis.shape[1] == 0:
        return vector

    gram = basis.T @ basis
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    _require_nonnegative_spectrum(eigenvalues, "low-rank inverse square-root Gram")
    positive = eigenvalues > 0

    if not torch.any(positive):
        return vector

    active_values = eigenvalues[positive]
    active_vectors = eigenvectors[:, positive]
    orthonormal = basis @ active_vectors / torch.sqrt(active_values).unsqueeze(0)
    coefficients = orthonormal.T @ vector
    scales = torch.rsqrt(1.0 + active_values) - 1.0
    result = vector + orthonormal @ (scales * coefficients)
    _require_finite_tensor(result, "low-rank inverse square-root core result")

    return result


def _ggn_derived_square_root_apply(
    execution: StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    jacobian, loss_hessian = _ggn_metric_factors(execution.batch, execution.params)
    loss_root = _psd_square_root(loss_hessian, "GGN-derived loss Hessian")

    if inverse:
        damping = _inverse_metric_damping(execution.operator)

        if damping <= 0.0:
            message = "GGN-derived inverse square root requires positive damping"
            raise MaterializationError(message)

        factor_basis = (loss_root @ jacobian).T
        diagonal = torch.full(
            (jacobian.shape[1],),
            damping,
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        flat_result = _low_rank_plus_diagonal_inverse_square_root_flat_apply(
            factor_basis,
            diagonal,
            _flatten_vector(execution.vector),
            adjoint=adjoint,
        )

        return _wrap_flat_vector(execution.params, flat_result)

    return _rectangular_square_root_apply(
        execution,
        adjoint=adjoint,
        name="GGN-derived square-root",
        widths=(jacobian.shape[1], jacobian.shape[0]),
        width_labels=("parameter width", "output width"),
        products=(
            lambda flat_vector: loss_root @ (jacobian @ flat_vector),
            lambda flat_vector: jacobian.T @ (loss_root @ flat_vector),
        ),
    )


def _psd_square_root(matrix: torch.Tensor, label: str) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    _require_nonnegative_spectrum(eigenvalues, label)

    result = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    _require_finite_tensor(result, f"{label} square root")

    return result


def _metric_square_root_factor_matrix(
    execution: StandardExecution,
    matrix: torch.Tensor,
    *,
    inverse: bool,
    path: str,
) -> torch.Tensor:
    if inverse:
        factor_matrix = torch.linalg.inv(
            _inverse_metric_matrix(execution.operator, matrix, execution.batch)
        )
    else:
        factor_matrix = matrix

    if path == SQRT_METRIC_CHOLESKY_PATH:
        _require_positive_definite_matrix(factor_matrix, "Cholesky square root")

        return torch.linalg.cholesky(factor_matrix)

    if path == SQRT_METRIC_EIGENBASIS_PATH:
        eigenvalues, eigenvectors = torch.linalg.eigh(factor_matrix)
        _require_positive_spectrum(eigenvalues, "eigenbasis square root")

        return eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

    message = f"metric square-root path is not lowered: {path}"
    raise MaterializationError(message)


def _lanczos_metric_square_root_product(
    execution: StandardExecution,
    matrix: torch.Tensor,
    vector: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    iterations, transform = _lanczos_sqrt_transform(execution, inverse=inverse)

    return _lanczos_matrix_function_product(matrix, vector, iterations, transform)


def _lanczos_matrix_free_metric_square_root_product(
    execution: StandardExecution,
    vector: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    iterations, transform = _lanczos_sqrt_transform(execution, inverse=inverse)

    def apply(flat_vector: torch.Tensor) -> torch.Tensor:
        return _metric_apply_flat(
            execution.operator,
            execution.batch,
            execution.vector,
            flat_vector,
            0.0,
            METRIC_STREAMING_PATH,
            execution.candidate.settings,
        )

    return _lanczos_matrix_function_product_from_apply(
        apply,
        vector,
        iterations,
        transform,
    )


def _lanczos_sqrt_transform(
    execution: StandardExecution,
    *,
    inverse: bool,
) -> tuple[int, Callable[[torch.Tensor], torch.Tensor]]:
    iterations = _sqrt_metric_lanczos_iterations(execution.candidate.settings)
    damping = _inverse_metric_damping(execution.operator) if inverse else 0.0

    def transform(values: torch.Tensor) -> torch.Tensor:
        shifted = values + damping

        if inverse:
            _require_positive_spectrum(shifted, "Lanczos square-root spectrum")

            return torch.rsqrt(shifted)

        _require_nonnegative_spectrum(shifted, "Lanczos square-root spectrum")

        return torch.sqrt(shifted)

    return iterations, transform


def _lanczos_matrix_function_product(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    iterations: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return _lanczos_matrix_function_product_from_apply(
        lambda flat_vector: matrix @ flat_vector,
        vector,
        iterations,
        transform,
    )


def _lanczos_matrix_function_product_from_apply(
    apply: Callable[[torch.Tensor], torch.Tensor],
    vector: torch.Tensor,
    iterations: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    norm = vector.norm()

    if torch.equal(norm, torch.zeros_like(norm)):
        return torch.zeros_like(vector)

    basis = []
    alphas = []
    betas = []
    current = vector / norm
    previous = torch.zeros_like(current)
    beta = torch.zeros((), dtype=vector.dtype, device=vector.device)

    for iteration in range(iterations):
        basis.append(current)
        residual = apply(current)
        alpha = current @ residual
        residual = residual - alpha * current - beta * previous
        beta = residual.norm()
        alphas.append(alpha)

        if torch.equal(beta, torch.zeros_like(beta)):
            break

        if iteration == iterations - 1:
            break

        betas.append(beta)
        previous = current
        current = residual / beta

    q_matrix = torch.stack(tuple(basis), dim=1)
    tri = _lanczos_tridiagonal(alphas, betas)
    eigenvalues, eigenvectors = torch.linalg.eigh(tri)
    first = torch.zeros(len(alphas), dtype=vector.dtype, device=vector.device)
    first[0] = norm
    projected = eigenvectors @ (transform(eigenvalues) * (eigenvectors.T @ first))

    return q_matrix @ projected


def _lanczos_tridiagonal(
    alphas: Sequence[torch.Tensor],
    betas: Sequence[torch.Tensor],
) -> torch.Tensor:
    tri = torch.diag(torch.stack(tuple(alphas)))

    if betas:
        off_diagonal = torch.stack(tuple(betas))
        tri = tri + torch.diag(off_diagonal, diagonal=1)
        tri = tri + torch.diag(off_diagonal, diagonal=-1)

    return tri


def _sqrt_metric_lanczos_iterations(settings: Mapping[str, Any]) -> int:
    value = settings.get("sqrt_metric.lanczos_iterations")

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = "sqrt_metric.lanczos_iterations must be a positive integer"
        raise MaterializationError(message)

    return value


def _require_positive_spectrum(values: torch.Tensor, label: str) -> None:
    if torch.any(values <= 0):
        message = f"{label} requires positive eigenvalues"
        raise MaterializationError(message)


def _require_nonnegative_spectrum(values: torch.Tensor, label: str) -> None:
    if torch.any(values < 0):
        message = f"{label} requires nonnegative eigenvalues"
        raise MaterializationError(message)


def _require_positive_definite_matrix(matrix: torch.Tensor, label: str) -> None:
    _, info = torch.linalg.cholesky_ex(matrix)

    if torch.any(info != 0):
        message = f"{label} requires positive definite matrix"
        raise MaterializationError(message)


def _require_metric_accumulation_settings(
    metric_path: str,
    settings: Mapping[str, Any],
) -> None:
    value = settings.get("metric.accumulation")

    if metric_path == METRIC_DENSE_PATH:
        if value is not None:
            message = "metric.accumulation applies only to non-dense metric paths"
            raise MaterializationError(message)

        return

    if value is None:
        message = "metric.accumulation is required for non-dense metric paths"
        raise MaterializationError(message)

    expected = (
        "streaming" if metric_path == METRIC_STREAMING_PATH else "materialized_blocks"
    )

    if value != expected:
        message = f"metric.accumulation must be {expected} for this path"
        raise MaterializationError(message)


def _metric_multiply_by_path(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    metric_path: str,
    settings: Mapping[str, Any],
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None = None,
) -> TensorTree:
    if _metric_representation_kind(operator) == "matrix_free":
        if metric_path != METRIC_STREAMING_PATH:
            message = "matrix_free metric requires streaming_multiply"
            raise MaterializationError(message)

        return _matrix_free_metric_multiply(
            operator,
            batch,
            vector,
            matrix_free_operators,
        )

    if metric_path == METRIC_FACTORIZED_PATH:
        return _factorized_metric_multiply(operator, batch, vector, settings)

    if metric_path == METRIC_BLOCKWISE_PATH:
        _require_metric_representation(operator, ("block_diagonal",))

        return _block_diagonal_metric_multiply(operator, batch, vector, settings)

    if metric_path == METRIC_STREAMING_PATH:
        return _streaming_metric_multiply(operator, batch, vector, settings)

    if metric_path == METRIC_DENSE_PATH:
        _require_metric_representation(operator, ("dense_matrix",))
        matrix = _metric_dense_matrix(operator, batch, vector)
        vector_tensor = _flatten_vector(vector)
        _require_finite_tensor(matrix, "metric matrix")
        _require_finite_tensor(vector_tensor, "metric vector")
        flat_result = _matmul_runtime(settings, matrix, vector_tensor)
        _require_finite_tensor(flat_result, "metric result")

        return _wrap_flat_vector(vector, flat_result)

    message = f"metric path is not lowered: {metric_path}"
    raise MaterializationError(message)


def _matrix_free_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None,
) -> TensorTree:
    product = _matrix_free_metric_product(operator)
    bindings = matrix_free_operators

    if bindings is None:
        bindings = MATRIX_FREE_RUNTIME_BINDINGS.get()

    selected = None if bindings is None else bindings.get(product)

    if selected is None:
        message = f"matrix_free metric requires selected sibling product: {product}"
        raise MaterializationError(message)

    result = selected(batch, vector)
    _require_finite_tree(result, "matrix-free metric result")

    return result


def _factorized_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=FACTORIZED_METRIC_MULTIPLY_BY_KIND,
        error_prefix="factorized metric path is not lowered for representation",
    )


def _streaming_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=STREAMING_METRIC_MULTIPLY_BY_KIND,
        error_prefix="metric streaming path is not lowered for representation",
    )


def _streaming_diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    diagonal = _metric_diagonal_tree(batch, vector)
    result = _tree_elementwise_mul_runtime(settings, diagonal, vector)
    _require_finite_tree(result, "streaming diagonal metric result")

    return result


def _streaming_block_diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    result = _block_diagonal_apply(
        _metric_blocks(batch),
        _flatten_vector(vector),
        settings,
        "streaming block metric result",
    )

    return _wrap_flat_vector(vector, result)


def _streaming_low_rank_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = _runtime_intermediate_tensor(_flatten_vector(vector), settings)
    basis, diagonal = _low_rank_factors(batch, vector)
    diagonal = _runtime_intermediate_tensor(diagonal, settings)
    result = _accumulation_tensor(diagonal, settings) * _accumulation_tensor(
        flat_vector, settings
    )

    for index in range(basis.shape[1]):
        column = _runtime_intermediate_tensor(basis[:, index], settings)
        projection = _dot_runtime(settings, column, flat_vector)
        result = result + _accumulation_tensor(column, settings) * projection

    _require_finite_tensor(result, "streaming low-rank metric result")

    return _wrap_flat_vector(vector, result)


def _streaming_kfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    factor_batch = _kfac_factor_batch(batch)
    vector_map = _kfac_vector_map(vector)
    result = {}

    for block in _kfac_blocks(operator):
        left = _kfac_factor(factor_batch, block.left_factor_key)
        right = _kfac_factor(factor_batch, block.right_factor_key)
        value = _kfac_vector_leaf(vector_map, block)
        _require_kfac_shapes(block, left, right, value)
        left_product = _matmul_runtime(settings, left, value)
        product = _matmul_runtime(settings, left_product, right.T)
        _require_finite_tensor(
            product,
            f"streaming KFAC metric result {block.parameter_name}",
        )
        result[block.parameter_name] = product

    return result


def _streaming_ggn_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = _runtime_intermediate_tensor(_flatten_vector(vector), settings)
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    output_vector = torch.stack(
        tuple(
            _dot_runtime(
                settings,
                _runtime_intermediate_tensor(row, settings),
                flat_vector,
            )
            for row in jacobian
        )
    )
    loss_vector = torch.stack(
        tuple(
            _dot_runtime(
                settings,
                _runtime_intermediate_tensor(row, settings),
                output_vector,
            )
            for row in loss_hessian
        )
    )
    result = torch.zeros_like(flat_vector)

    for row, weight in zip(jacobian, loss_vector, strict=True):
        result = result + (
            _accumulation_tensor(_runtime_intermediate_tensor(row, settings), settings)
            * _accumulation_tensor(weight, settings)
        )

    _require_finite_tensor(result, "streaming GGN-derived metric result")

    return _wrap_flat_vector(vector, result)


MetricMultiplyRunner = Callable[
    [OperatorSpec, Batch, TensorTree, Mapping[str, Any]],
    TensorTree,
]
InverseMetricBatchRunner = Callable[[StandardExecution], TensorTree]


def _metric_multiply_by_kind(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    runners: Mapping[str, MetricMultiplyRunner],
    error_prefix: str,
) -> TensorTree:
    kind = _metric_representation_kind(operator)
    runner = runners.get(kind)

    if runner is None:
        message = f"{error_prefix}: {kind}"
        raise MaterializationError(message)

    return runner(operator, batch, vector, settings)


FACTORIZED_METRIC_MULTIPLY_BY_KIND = {
    "diagonal_tree": _diagonal_metric_multiply,
    "low_rank_factors": _low_rank_metric_multiply,
    "kfac_factors": _kfac_metric_multiply,
    "ekfac_factors": _ekfac_metric_multiply,
    "ggn_derived_factors": _ggn_metric_multiply,
}
FACTORIZED_INVERSE_METRIC_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply,
    "low_rank_factors": _low_rank_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
    "ekfac_factors": _ekfac_inverse_metric_multiply,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply,
}
INVERSE_METRIC_MULTIPLY_BY_PATH = {
    INVERSE_METRIC_FACTORIZED_PATH: (
        FACTORIZED_INVERSE_METRIC_BY_KIND,
        "factorized inverse path is not lowered for representation",
    ),
    INVERSE_METRIC_BLOCKWISE_PATH: (
        {"block_diagonal": _block_diagonal_inverse_metric_multiply},
        "metric representation kind is not supported by path",
    ),
    INVERSE_METRIC_WOODBURY_PATH: (
        {"low_rank_factors": _low_rank_inverse_metric_multiply},
        "metric representation kind is not supported by path",
    ),
}
BLOCK_PRECONDITIONER_BY_KIND = {
    "block_diagonal": _block_diagonal_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
}
FACTORIZED_PRECONDITIONER_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply,
    "low_rank_factors": _low_rank_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply,
}
FACTORIZED_INVERSE_METRIC_BATCH_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply_batch,
    "low_rank_factors": _low_rank_inverse_metric_multiply_batch,
    "kfac_factors": _kfac_inverse_metric_multiply_batch,
    "ekfac_factors": _ekfac_inverse_metric_multiply_batch,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply_batch,
}
STREAMING_METRIC_MULTIPLY_BY_KIND = {
    "diagonal_tree": _streaming_diagonal_metric_multiply,
    "block_diagonal": _streaming_block_diagonal_metric_multiply,
    "low_rank_factors": _streaming_low_rank_metric_multiply,
    "kfac_factors": _streaming_kfac_metric_multiply,
    "ekfac_factors": _ekfac_metric_multiply,
    "ggn_derived_factors": _streaming_ggn_metric_multiply,
}


def _run_inverse_metric(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            INVERSE_METRIC_DENSE_PATH,
            INVERSE_METRIC_CG_PATH,
            INVERSE_METRIC_CHOLESKY_PATH,
            INVERSE_METRIC_EIGH_PATH,
            INVERSE_METRIC_SVD_PATH,
            INVERSE_METRIC_FACTORIZED_PATH,
            INVERSE_METRIC_BLOCKWISE_PATH,
            INVERSE_METRIC_WOODBURY_PATH,
        ),
    )

    if execution.compiled_inner is None:
        result = _run_inverse_metric_by_mode(execution)
    else:
        result = execution.compiled_inner()

    _require_finite_tree(result, "inverse metric result")

    return result


def _run_inverse_metric_by_mode(execution: StandardExecution) -> TensorTree:
    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "single_loop":
        return _run_inverse_metric_vector_single_loop(execution)

    if mode == "manual_batch":
        return _run_vector_manual_batches(
            execution,
            _run_inverse_metric_vector_single_loop,
        )

    return _inverse_metric_solve_by_path(execution)


def _run_inverse_metric_vector_single_loop(
    execution: StandardExecution,
) -> TensorTree:
    settings = execution.candidate.settings

    if (
        settings.get("inverse_metric.multi_rhs") == "block"
        or settings.get("inverse_metric.factor_reuse") == "reuse_factor_across_rhs"
    ):
        return _run_inverse_metric_rhs_batch(execution)

    return _run_vector_single_loop(execution, _inverse_metric_solve_by_path)


def _run_inverse_metric_rhs_batch(
    execution: StandardExecution,
) -> TensorTree:
    if execution.path not in INVERSE_METRIC_FACTOR_REUSE_PATHS:
        message = "block inverse metric RHS requires a batched solve path"
        raise MaterializationError(message)

    if execution.path in INVERSE_METRIC_DIRECT_SOLVE_PATHS:
        return _run_inverse_metric_reused_dense_factor_batch(execution)

    if execution.path == INVERSE_METRIC_CG_PATH:
        return _conjugate_gradient_inverse_metric_multiply_batch(execution)

    if execution.path == INVERSE_METRIC_FACTORIZED_PATH:
        return _factorized_inverse_metric_multiply_batch(execution)

    if execution.path == INVERSE_METRIC_BLOCKWISE_PATH:
        _require_metric_representation(execution.operator, ("block_diagonal",))

        return _block_diagonal_inverse_metric_multiply_batch(execution)

    _require_metric_representation(execution.operator, ("low_rank_factors",))

    return _low_rank_inverse_metric_multiply_batch(execution)


def _run_inverse_metric_reused_dense_factor_batch(
    execution: StandardExecution,
) -> TensorTree:
    _require_metric_representation(execution.operator, ("dense_matrix",))
    inverse_matrix = _inverse_metric_matrix(
        execution.operator,
        _metric_dense_matrix(execution.operator, execution.batch, execution.vector),
        execution.batch,
    )
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    _require_finite_tensor(inverse_matrix, "metric matrix")
    _require_finite_tensor(vector_batch, "inverse metric vector batch")
    result = _dense_inverse_metric_solve_batch(
        inverse_matrix,
        vector_batch,
        execution.path,
    )
    _require_finite_tensor(result, "inverse metric batched result")

    return _wrap_flat_vector_batch(execution.params, result)


def _factorized_inverse_metric_multiply_batch(
    execution: StandardExecution,
) -> TensorTree:
    kind = _metric_representation_kind(execution.operator)
    runner = FACTORIZED_INVERSE_METRIC_BATCH_BY_KIND.get(kind)

    if runner is not None:
        return runner(execution)

    message = f"factorized inverse batch path is not lowered for representation: {kind}"
    raise MaterializationError(message)


def _flat_inverse_metric_vector_batch(execution: StandardExecution) -> torch.Tensor:
    vector_in_dims = _vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )

    return _flatten_vector_batch(execution.params, execution.vector, vector_in_dims)


def _inverse_metric_solve_by_path(execution: StandardExecution) -> TensorTree:
    if execution.path == INVERSE_METRIC_CG_PATH:
        return _conjugate_gradient_inverse_metric_multiply(execution)

    runner_row = INVERSE_METRIC_MULTIPLY_BY_PATH.get(execution.path)

    if runner_row is not None:
        runners, error_prefix = runner_row

        return _metric_multiply_by_kind(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.candidate.settings,
            runners=runners,
            error_prefix=error_prefix,
        )

    _require_metric_representation(execution.operator, ("dense_matrix",))
    inverse_matrix = _inverse_metric_matrix(
        execution.operator,
        _metric_dense_matrix(execution.operator, execution.batch, execution.vector),
        execution.batch,
    )
    vector_tensor = _flatten_vector(execution.vector)
    _require_finite_tensor(inverse_matrix, "metric matrix")
    _require_finite_tensor(vector_tensor, "inverse metric vector")
    result = _dense_inverse_metric_solve(
        inverse_matrix,
        vector_tensor,
        execution.path,
    )
    _require_finite_tensor(result, "inverse metric result")

    return _wrap_flat_vector(execution.vector, result)


def _conjugate_gradient_inverse_metric_multiply(
    execution: StandardExecution,
) -> TensorTree:
    budget = _inverse_metric_iteration_budget(execution.candidate.settings)
    preconditioner = _inverse_metric_preconditioner(execution.candidate.settings)
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)
    _require_metric_accumulation_settings(metric_path, execution.candidate.settings)
    damping = _inverse_metric_damping(execution.operator)
    _require_positive_matrix_free_damping(execution.operator, damping)
    tolerance = _inverse_metric_tolerance(execution.operator)
    solution = _conjugate_gradient_inverse_metric_batch_solve(
        execution,
        execution.vector,
        _flatten_vector(execution.vector).unsqueeze(0),
        budget,
        preconditioner,
        metric_path,
        damping,
        tolerance,
    )
    result = solution[0]

    _require_finite_tensor(result, "conjugate gradient result")

    return _wrap_flat_vector(execution.vector, result)


def _zero_numerator_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        numerator == 0,
        torch.zeros_like(numerator),
        numerator / denominator,
    )


def _require_positive_matrix_free_damping(
    operator: OperatorSpec,
    damping: float,
) -> None:
    if _metric_representation_kind(operator) != "matrix_free":
        return

    if damping > 0.0:
        return

    message = "matrix_free conjugate_gradient requires positive damping"
    raise MaterializationError(message)


def _conjugate_gradient_inverse_metric_multiply_batch(
    execution: StandardExecution,
) -> TensorTree:
    budget = _inverse_metric_iteration_budget(execution.candidate.settings)
    preconditioner = _inverse_metric_preconditioner(execution.candidate.settings)
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)
    _require_metric_accumulation_settings(metric_path, execution.candidate.settings)
    damping = _inverse_metric_damping(execution.operator)
    _require_positive_matrix_free_damping(execution.operator, damping)
    tolerance = _inverse_metric_tolerance(execution.operator)
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    solution = _conjugate_gradient_inverse_metric_batch_solve(
        execution,
        execution.params,
        vector_batch,
        budget,
        preconditioner,
        metric_path,
        damping,
        tolerance,
    )
    _require_finite_tensor(solution, "batched conjugate gradient result")

    return _wrap_flat_vector_batch(execution.params, solution)


def _conjugate_gradient_inverse_metric_batch_solve(
    execution: StandardExecution,
    template: TensorTree,
    vectors: torch.Tensor,
    budget: int,
    preconditioner: str,
    metric_path: str,
    damping: float,
    tolerance: float | None,
) -> torch.Tensor:
    solution = torch.zeros_like(vectors)
    residual = vectors - _metric_apply_flat_batch(
        execution.operator,
        execution.batch,
        template,
        solution,
        damping,
        metric_path,
        execution.candidate.settings,
    )

    if _batched_cg_residual_satisfies_tolerance(residual, vectors, tolerance):
        return solution

    preconditioned = _apply_inverse_metric_preconditioner_batch(
        execution.operator,
        execution.batch,
        template,
        residual,
        preconditioner,
        execution.candidate.settings,
    )
    direction = preconditioned
    residual_dot = _batched_dot_runtime(
        execution.candidate.settings,
        residual,
        preconditioned,
    )

    for _ in range(budget):
        matrix_direction = _metric_apply_flat_batch(
            execution.operator,
            execution.batch,
            template,
            direction,
            damping,
            metric_path,
            execution.candidate.settings,
        )
        step = _zero_numerator_divide(
            residual_dot,
            _batched_dot_runtime(
                execution.candidate.settings,
                direction,
                matrix_direction,
            ),
        )
        solution = solution + step[:, None] * direction
        residual = residual - step[:, None] * matrix_direction

        if _batched_cg_residual_satisfies_tolerance(residual, vectors, tolerance):
            break

        preconditioned = _apply_inverse_metric_preconditioner_batch(
            execution.operator,
            execution.batch,
            template,
            residual,
            preconditioner,
            execution.candidate.settings,
        )
        next_residual_dot = _batched_dot_runtime(
            execution.candidate.settings,
            residual,
            preconditioned,
        )
        direction = preconditioned + _zero_numerator_divide(
            next_residual_dot,
            residual_dot,
        )[:, None] * (direction)
        residual_dot = next_residual_dot

    _require_finite_tensor(solution, "batched conjugate gradient result")

    return solution


def _batched_cg_residual_satisfies_tolerance(
    residual: torch.Tensor,
    right_hand_sides: torch.Tensor,
    tolerance: float | None,
) -> bool:
    if tolerance is None:
        return False

    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    denominator = torch.linalg.vector_norm(right_hand_sides, dim=1)
    scaled_residual = torch.where(
        denominator == 0,
        residual_norm,
        residual_norm / denominator,
    )

    return bool(torch.all(scaled_residual <= tolerance).item())


def _metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return _runtime_path_from_settings(
        settings,
        "metric",
        "metric.multiply_path",
    )


def _inverse_metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return _runtime_path_from_settings(
        settings,
        "inverse_metric",
        "inverse_metric.solve_path",
    )


def _metric_apply_flat(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    flat_vector: torch.Tensor,
    damping: float,
    metric_path: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    vector = _wrap_flat_vector(template, flat_vector)
    result = _metric_multiply_by_path(operator, batch, vector, metric_path, settings)
    flat_result = _flatten_vector(result) + damping * flat_vector
    _require_finite_tensor(flat_result, "metric apply result")

    return flat_result


def _metric_apply_flat_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    flat_batch: torch.Tensor,
    damping: float,
    metric_path: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    parts = [
        _metric_apply_flat(
            operator,
            batch,
            template,
            flat_vector,
            damping,
            metric_path,
            settings,
        )
        for flat_vector in flat_batch
    ]

    result = torch.stack(tuple(parts), dim=0)
    _require_finite_tensor(result, "batched metric apply result")

    return result


def _apply_inverse_metric_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    residual: torch.Tensor,
    preconditioner: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    residual_tree = _wrap_flat_vector(template, residual)

    if preconditioner == "none":
        result = residual
    elif preconditioner == "diagonal":
        diagonal = torch.diag(
            _inverse_metric_matrix(
                operator,
                _metric_dense_matrix(operator, batch, template),
                batch,
            )
        )
        result = residual / diagonal
    elif preconditioner == "block_diagonal":
        result = _flatten_vector(
            _block_or_kfac_preconditioner(operator, batch, residual_tree)
        )
    elif preconditioner == "factorized_metric":
        result = _flatten_vector(
            _factorized_metric_preconditioner(operator, batch, residual_tree, settings)
        )
    else:
        message = f"inverse metric preconditioner is unsupported: {preconditioner}"
        raise MaterializationError(message)

    _require_finite_tensor(result, "inverse metric preconditioner result")

    return result


def _apply_inverse_metric_preconditioner_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    residual_batch: torch.Tensor,
    preconditioner: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    parts = [
        _apply_inverse_metric_preconditioner(
            operator,
            batch,
            template,
            residual,
            preconditioner,
            settings,
        )
        for residual in residual_batch
    ]

    result = torch.stack(tuple(parts), dim=0)
    _require_finite_tensor(result, "batched inverse metric preconditioner result")

    return result


def _block_or_kfac_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        {},
        runners=BLOCK_PRECONDITIONER_BY_KIND,
        error_prefix="block preconditioner is not lowered for representation",
    )


def _factorized_metric_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=FACTORIZED_PRECONDITIONER_BY_KIND,
        error_prefix="factorized preconditioner is not lowered for representation",
    )


def _inverse_metric_iteration_budget(settings: Mapping[str, Any]) -> int:
    value = settings.get("inverse_metric.iteration_budget")

    if not isinstance(value, int) or value < 1:
        message = "inverse_metric.iteration_budget must be a positive integer"
        raise MaterializationError(message)

    return value


def _inverse_metric_preconditioner(settings: Mapping[str, Any]) -> str:
    value = settings.get("inverse_metric.preconditioner")

    if not isinstance(value, str):
        message = "inverse_metric.preconditioner is required"
        raise MaterializationError(message)

    if value == "matrix_free":
        message = (
            "inverse_metric.preconditioner=matrix_free requires named sibling "
            "product lowering"
        )
        raise MaterializationError(message)

    if value not in {"none", "diagonal", "block_diagonal", "factorized_metric"}:
        message = f"inverse_metric.preconditioner is unsupported: {value}"
        raise MaterializationError(message)

    return value


def _dense_inverse_metric_solve(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    path: str,
) -> torch.Tensor:
    flat_vector = vector.reshape(-1)
    result = _dense_inverse_metric_solve_batch(
        matrix,
        flat_vector.unsqueeze(0),
        path,
    )[0]

    return result.reshape_as(vector)


def _dense_inverse_metric_solve_batch(
    matrix: torch.Tensor,
    vector_batch: torch.Tensor,
    path: str,
) -> torch.Tensor:
    if vector_batch.ndim != MATRIX_DIMS:
        message = "batched inverse metric vectors must flatten to a matrix"
        raise MaterializationError(message)

    rhs = vector_batch.T

    if path == INVERSE_METRIC_DENSE_PATH:
        return torch.linalg.solve(matrix, rhs).T

    if path == INVERSE_METRIC_CHOLESKY_PATH:
        factor = torch.linalg.cholesky(matrix)

        return torch.cholesky_solve(rhs, factor).T

    if path == INVERSE_METRIC_EIGH_PATH:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        coefficients = eigenvectors.T @ rhs

        return (eigenvectors @ (coefficients / eigenvalues[:, None])).T

    if path == INVERSE_METRIC_SVD_PATH:
        left, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
        coefficients = left.T @ rhs

        return (right_h.T @ (coefficients / singular_values[:, None])).T

    message = f"batched inverse metric solve path is not lowered: {path}"
    raise MaterializationError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class StandardMetricOperator:
    """Materialized metric selected by the standard runtime."""

    candidate: Candidate
    record: FullSizeRecord
    operator: OperatorSpec
    representation: Mapping[str, Any]
    default_operation: str = "multiply"
    damping: float | Mapping[str, float] = 0.0
    inverse_path: str | None = None
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]] = (
        dataclasses.field(default_factory=dict)
    )

    def __call__(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return the selected metric-vector product.

        Raises:
            MaterializationError: If the default metric operation is unsupported.
        """
        if self.default_operation == "multiply":
            return self.multiply(batch, vector)

        if self.default_operation == "inverse_multiply":
            return self.inverse_multiply(batch, vector)

        message = f"unsupported metric default operation: {self.default_operation}"
        raise MaterializationError(message)

    def multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return metric-vector product."""
        return self._run_metric_runtime(batch, vector, self._multiply_runtime)

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse metric-vector product."""
        return self._run_metric_runtime(batch, vector, self._inverse_runtime)

    def _run_metric_runtime(
        self,
        batch: Batch,
        vector: TensorTree,
        operation: Callable[[Batch, TensorTree], TensorTree],
    ) -> TensorTree:
        return _run_with_backend_settings(
            self.candidate.settings,
            lambda: operation(*self._runtime_inputs(batch, vector)),
        )

    def _multiply_runtime(self, batch: Batch, vector: TensorTree) -> TensorTree:
        metric_operator = self._operator_spec("metric")
        path = _runtime_path(metric_operator, self.candidate)
        result = _metric_multiply_by_path(
            metric_operator,
            batch,
            vector,
            path,
            self.candidate.settings,
            self.matrix_free_operators,
        )
        _require_finite_tree(result, "metric result")

        return result

    def _inverse_runtime(self, batch: Batch, vector: TensorTree) -> TensorTree:
        if self.inverse_path is None:
            message = "inverse_multiply requires an inverse_metric selection"
            raise MaterializationError(message)

        inverse_operator = self._operator_for_inverse()

        if self.inverse_path == INVERSE_METRIC_CG_PATH:
            execution = StandardExecution(
                inverse_operator,
                self.candidate,
                self.inverse_path,
                batch,
                vector,
                {},
                {},
                None,
                ObjectiveContext(
                    family=self.record.family,
                    candidate_id=self.candidate.candidate_id,
                    settings=dict(self.candidate.settings),
                ),
                {},
                {},
            )
            result = _run_with_matrix_free_runtime_bindings(
                self.matrix_free_operators,
                lambda: _conjugate_gradient_inverse_metric_multiply(execution),
            )
        elif self.inverse_path in INVERSE_METRIC_MULTIPLY_BY_PATH:
            runners, error_prefix = INVERSE_METRIC_MULTIPLY_BY_PATH[self.inverse_path]
            result = _metric_multiply_by_kind(
                inverse_operator,
                batch,
                vector,
                self.candidate.settings,
                runners=runners,
                error_prefix=error_prefix,
            )
        else:
            _require_metric_representation(inverse_operator, ("dense_matrix",))
            matrix = self._dense_matrix(batch, vector)
            inverse_matrix = _inverse_metric_matrix(inverse_operator, matrix, batch)
            flat_result = _dense_inverse_metric_solve(
                inverse_matrix,
                _flatten_vector(vector),
                self.inverse_path,
            )
            _require_finite_tensor(flat_result, "inverse metric result")
            result = _wrap_flat_vector(vector, flat_result)

        _require_finite_tree(result, "inverse metric result")

        return result

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return metric inner product."""

        def callback() -> torch.Tensor:
            runtime_batch, runtime_left = self._runtime_inputs(batch, left)
            runtime_right = _runtime_vector(right, self.candidate.settings)
            left_tensor = _flatten_vector(runtime_left)
            _require_finite_tensor(left_tensor, "metric inner left vector")
            metric_right = self.multiply(runtime_batch, runtime_right)
            result = _tree_dot_runtime(
                self.candidate.settings,
                runtime_left,
                metric_right,
            )
            _require_finite_tensor(result, "metric inner result")

            return result

        return _run_with_backend_settings(self.candidate.settings, callback)

    def _runtime_inputs(
        self,
        batch: Batch,
        vector: TensorTree,
    ) -> tuple[Batch, TensorTree]:
        runtime_batch = _runtime_batch(
            batch,
            self.candidate.settings,
            mmap_residency=self.mmap_residency,
        )
        runtime_vector = _runtime_vector(
            vector,
            self.candidate.settings,
            mmap_residency=self.mmap_residency,
        )
        vector_tensor = _flatten_vector(runtime_vector)
        _require_finite_tensor(vector_tensor, "metric vector")

        return runtime_batch, runtime_vector

    def _dense_matrix(self, batch: Batch, vector: TensorTree) -> torch.Tensor:
        operator = self._operator_spec("metric")

        return _metric_dense_matrix(operator, batch, vector)

    def _operator_for_inverse(self) -> OperatorSpec:
        damping_kind = _inverse_metric_damping_kind(self.operator)
        damping_value = _inverse_metric_damping_payload(self.operator)

        return self._operator_spec(
            "inverse_metric",
            {
                "damping": damping_value,
                "damping_kind": damping_kind,
                "damping_value": damping_value,
            },
        )

    def _operator_spec(
        self,
        kind: str,
        semantics: Mapping[str, Any] | None = None,
    ) -> OperatorSpec:
        full_semantics = {"representation": dict(self.representation)}

        if semantics is not None:
            full_semantics.update(semantics)

        return dataclasses.replace(
            self.operator,
            kind=kind,
            semantics=full_semantics,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricBlock:
    """One Kronecker-factored metric block for a matrix parameter."""

    parameter_name: str
    left_factor_key: str
    right_factor_key: str


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricOperator:
    """Metric operations backed by Kronecker-factored blocks."""

    blocks: tuple[KFACMetricBlock, ...]
    damping: float | Mapping[str, float] = 0.0
    damping_kind: str = "scalar"
    damping_policy: str | None = None
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __call__(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return metric-vector product."""
        return self.multiply(batch, vector)

    def multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return KFAC metric-vector product."""
        vector_map = _kfac_vector_map(vector)
        result = {}

        for block in self.blocks:
            left = _kfac_factor(batch, block.left_factor_key)
            right = _kfac_factor(batch, block.right_factor_key)
            value = _kfac_vector_leaf(vector_map, block)
            _require_kfac_shapes(block, left, right, value)
            product = _matmul_runtime(
                self.settings,
                _matmul_runtime(self.settings, left, value),
                right.T,
            )
            _require_finite_tensor(
                product, f"KFAC metric result {block.parameter_name}"
            )
            result[block.parameter_name] = product

        return result

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse KFAC metric-vector product."""
        vector_map = _kfac_vector_map(vector)
        result = {}

        for block in self.blocks:
            left = _kfac_factor(batch, block.left_factor_key)
            right = _kfac_factor(batch, block.right_factor_key)
            value = _kfac_vector_leaf(vector_map, block)
            _require_kfac_shapes(block, left, right, value)
            damping = _resolved_group_damping(
                self.damping,
                self.damping_kind,
                block.parameter_name,
            )
            product = _kfac_inverse_product(
                left,
                right,
                value,
                damping,
                _resolved_group_damping_kind(self.damping_kind),
                self.damping_policy,
            )
            _require_finite_tensor(
                product,
                f"inverse KFAC metric result {block.parameter_name}",
            )
            result[block.parameter_name] = product

        return result

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return KFAC metric inner product."""
        return _tree_dot_runtime(self.settings, left, self.multiply(batch, right))


def _kfac_vector_map(vector: TensorTree) -> dict[str, TensorTree]:
    if type(vector) is not dict:
        message = "KFAC vector must be a tensor-tree mapping"
        raise MaterializationError(message)

    return dict(vector)


def _kfac_factor(batch: Batch, key: str) -> torch.Tensor:
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC factor is missing or not a tensor: {key}"
        raise MaterializationError(message)

    _require_finite_tensor(value, f"KFAC factor {key}")

    if value.ndim != MATRIX_DIMS or value.shape[0] != value.shape[1]:
        message = f"KFAC factor must be square: {key}"
        raise MaterializationError(message)

    return value


def _kfac_inverse_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    damping: float,
    damping_kind: str,
    damping_policy: str | None,
) -> torch.Tensor:
    return _kfac_inverse_product_with_damping(
        left,
        right,
        value,
        damping,
        damping_kind,
        damping_policy,
    )


def _kfac_inverse_product_batch(
    operator: OperatorSpec,
    parameter_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    damping_kind = _inverse_metric_damping_kind(operator)
    damping = _resolved_group_damping(
        _inverse_metric_damping_payload(operator),
        damping_kind,
        parameter_name,
    )
    damping_kind = _resolved_group_damping_kind(damping_kind)
    damping_policy = _inverse_metric_damping_policy(operator)

    return _kfac_inverse_product_with_damping(
        left,
        right,
        value,
        damping,
        damping_kind,
        damping_policy,
    )


def _kfac_inverse_product_with_damping(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    damping: float,
    damping_kind: str,
    damping_policy: str | None,
) -> torch.Tensor:
    if damping_kind == "kfac_pi":
        left_shift, right_shift = _kfac_pi_shifts(
            left,
            right,
            damping,
            damping_policy,
        )

        return _kfac_factored_inverse_product(
            left
            + left_shift
            * torch.eye(
                left.shape[0],
                dtype=left.dtype,
                device=left.device,
            ),
            right
            + right_shift
            * torch.eye(
                right.shape[0],
                dtype=right.dtype,
                device=right.device,
            ),
            value,
        )

    if damping_kind != "scalar":
        message = f"KFAC inverse does not lower damping kind: {damping_kind}"
        raise MaterializationError(message)

    if damping <= 0.0:
        return _kfac_factored_inverse_product(left, right, value)

    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    rotated = left_eigenvectors.T @ value @ right_eigenvectors
    denominator = left_eigenvalues[:, None] * right_eigenvalues[None, :] + damping
    solved = rotated / denominator

    return left_eigenvectors @ solved @ right_eigenvectors.T


def _kfac_factored_inverse_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_solved = torch.linalg.solve(left, value)

    return torch.linalg.solve(right, left_solved.mT).mT


def _kfac_pi_shifts(
    left: torch.Tensor,
    right: torch.Tensor,
    damping: float,
    policy: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = torch.sqrt(left.new_tensor(damping))

    if policy == "equal":
        return root, root

    if policy != "trace_norm":
        message = f"kfac_pi policy is unsupported: {policy}"
        raise MaterializationError(message)

    left_mean = torch.trace(left) / left.shape[0]
    right_mean = torch.trace(right) / right.shape[0]
    _require_positive_spectrum(
        torch.stack((left_mean, right_mean)),
        "KFAC pi factor traces",
    )
    ratio = torch.sqrt(left_mean / right_mean)

    return ratio * root, root / ratio


def _kfac_batched_vector_leaf(
    vector: Mapping[str, TensorTree],
    in_dims: Mapping[str, Any],
    block: KFACMetricBlock,
    left: torch.Tensor,
    right: torch.Tensor,
    vector_count: int,
) -> torch.Tensor:
    value = vector.get(block.parameter_name)
    in_dim = in_dims.get(block.parameter_name)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC vector leaf is missing or not a tensor: {block.parameter_name}"
        raise MaterializationError(message)

    _require_finite_tensor(value, f"KFAC vector {block.parameter_name}")

    if in_dim is None:
        _require_kfac_shapes(block, left, right, value)

        return value.expand(vector_count, *value.shape)

    if not isinstance(in_dim, int) or isinstance(in_dim, bool):
        message = "KFAC vectorization.in_dims values must be integers or None"
        raise MaterializationError(message)

    dim = _normalized_vector_dim(value, in_dim)
    expected = (left.shape[0], right.shape[0])
    unbatched_shape = value.shape[:dim] + value.shape[dim + 1 :]

    if tuple(unbatched_shape) != expected:
        message = (
            f"KFAC vector leaf shape mismatch for {block.parameter_name}: "
            f"{tuple(unbatched_shape)} != {expected}"
        )
        raise MaterializationError(message)

    if value.shape[dim] != vector_count:
        message = "KFAC vectorized dimensions differ"
        raise MaterializationError(message)

    return value.movedim(dim, 0)


def _kfac_vector_leaf(
    vector: Mapping[str, TensorTree],
    block: KFACMetricBlock,
) -> torch.Tensor:
    value = vector.get(block.parameter_name)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC vector leaf is missing or not a tensor: {block.parameter_name}"
        raise MaterializationError(message)

    _require_finite_tensor(value, f"KFAC vector {block.parameter_name}")

    if value.ndim != MATRIX_DIMS:
        message = f"KFAC vector leaf must be a matrix: {block.parameter_name}"
        raise MaterializationError(message)

    return value


def _require_kfac_shapes(
    block: KFACMetricBlock,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> None:
    expected = (left.shape[0], right.shape[0])

    if tuple(value.shape) != expected:
        message = (
            f"KFAC vector leaf shape mismatch for {block.parameter_name}: "
            f"{tuple(value.shape)} != {expected}"
        )
        raise MaterializationError(message)


STANDARD_RUNNERS = {
    "gradient": _run_gradient,
    "jvp": _run_jvp,
    "vjp": _run_vjp,
    "hvp": _run_hvp,
    "ggnvp": _run_ggnvp,
    "fisher_vp": _run_fisher_vp,
    "sampled_fisher_vp": _run_sampled_fisher_vp,
    "empirical_fisher_vp": _run_empirical_fisher_vp,
    "per_example_gradient": _run_per_example_gradient,
    "metric": _run_metric,
    "sqrt_metric": _run_sqrt_metric,
    "inverse_sqrt_metric": _run_sqrt_metric,
    "metric_inner": _run_metric_inner,
    "inverse_metric": _run_inverse_metric,
    "inverse_metric_inner": _run_inverse_metric_inner,
}


def _standard_materializer(
    operation_factory: RuntimeOperationFactory,
    operator: OperatorSpec | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> Materializer:
    mmap_residency_callback = mmap_residency

    def callback(candidate: Candidate, record: FullSizeRecord) -> Any:
        if (
            record.family != candidate.family
            or record.candidate_id != candidate.candidate_id
        ):
            message = "selected record does not match selected candidate"
            raise MaterializationError(message)

        if operator is not None and operator.kind == "metric":
            return StandardMetricOperator(
                candidate,
                record,
                operator,
                _metric_representation(operator),
                mmap_residency=mmap_residency_callback,
            )

        if operator is not None and operator.kind == "inverse_metric":
            return StandardMetricOperator(
                candidate,
                record,
                operator,
                _metric_representation(operator),
                default_operation="inverse_multiply",
                damping=_inverse_metric_damping_payload(operator),
                inverse_path=_runtime_path(operator, candidate),
                mmap_residency=mmap_residency_callback,
            )

        def selected(batch: Batch, vector: TensorTree) -> TensorTree:
            return operation_factory(candidate, batch, vector)()

        return selected

    return CallableMaterializer(
        "vptune.standard_runtime",
        PACKAGE_VERSION,
        {"operation_factory": dict(operation_factory.identity())},
        {"callback": "_standard_materializer.callback"},
        callback,
    )


def standard_runtime_with_matrix_free_bindings(
    runtime: RuntimeConfig,
    *,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    binding_signature: Mapping[str, Any],
) -> RuntimeConfig:
    """Return a standard runtime bound to selected matrix-free products."""
    if not bindings:
        return runtime

    binding_map = dict(bindings)
    runtime_signature = {
        **dict(runtime.signature),
        "matrix_free_bindings": dict(binding_signature),
    }
    operation_factory = _matrix_free_bound_operation_factory(
        runtime.operation_factory,
        binding_map,
        runtime_signature,
    )
    reference_check = _matrix_free_bound_reference_check(
        runtime.reference_check,
        binding_map,
        runtime_signature,
    )
    materializer = _matrix_free_bound_materializer(
        runtime.materializer,
        binding_map,
        runtime_signature,
    )

    return dataclasses.replace(
        runtime,
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        signature=runtime_signature,
    )


def _matrix_free_bound_operation_factory(
    operation_factory: RuntimeOperationFactory,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> RuntimeOperationFactory:
    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        operation = operation_factory(candidate, batch, vector)

        def bound_operation() -> TensorTree:
            return _run_with_matrix_free_runtime_bindings(bindings, operation)

        return bound_operation

    return CallableOperationFactory(
        "vptune.matrix_free_bound_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(operation_factory.identity())},
        factory,
    )


def _matrix_free_bound_reference_check(
    reference_check: RuntimeReferenceCheck,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> RuntimeReferenceCheck:
    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        return _run_with_matrix_free_runtime_bindings(
            bindings,
            lambda: reference_check(candidate, batch, vector),
        )

    return CallableReferenceCheck(
        "vptune.matrix_free_bound_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(reference_check.identity())},
        check,
    )


def _matrix_free_bound_materializer(
    materializer: Materializer,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> Materializer:
    def callback(candidate: Candidate, record: FullSizeRecord) -> Any:
        selected = materializer(candidate, record)

        if isinstance(selected, StandardMetricOperator):
            return dataclasses.replace(selected, matrix_free_operators=dict(bindings))

        if callable(selected):

            def bound_selected(batch: Batch, vector: TensorTree) -> TensorTree:
                return _run_with_matrix_free_runtime_bindings(
                    bindings,
                    lambda: selected(batch, vector),
                )

            return bound_selected

        return selected

    return CallableMaterializer(
        "vptune.matrix_free_bound_materializer",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(materializer.identity())},
        callback,
    )


def _run_with_matrix_free_runtime_bindings(
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    callback: Callable[[], Any],
) -> Any:
    token = MATRIX_FREE_RUNTIME_BINDINGS.set(dict(bindings))

    try:
        return callback()
    finally:
        MATRIX_FREE_RUNTIME_BINDINGS.reset(token)


def _runtime_path(operator: OperatorSpec, candidate: Candidate) -> str:
    spec_path = _spec_runtime_path(operator, candidate)

    if spec_path is not None:
        if "operator_path" in candidate.settings:
            message = "candidate cannot mix operator_path with SPEC path keys"
            raise MaterializationError(message)

        return spec_path

    if "operator_path" in candidate.settings:
        message = f"operator_path is not a setting for {operator.kind}"
        raise MaterializationError(message)

    if operator.kind in SPEC_REQUIRED_PATH_OPERATORS:
        key = SPEC_PATH_KEYS[operator.kind]
        message = f"{key} is required for {operator.kind}"
        raise MaterializationError(message)

    message = f"standard runtime has no SPEC path key for {operator.kind}"
    raise MaterializationError(message)


def _spec_runtime_path(operator: OperatorSpec, candidate: Candidate) -> str | None:
    special_paths = {
        "ggnvp": _ggn_spec_runtime_path,
        "fisher_vp": _fisher_spec_runtime_path,
        "sampled_fisher_vp": _sampled_fisher_spec_runtime_path,
        "empirical_fisher_vp": _empirical_fisher_spec_runtime_path,
        "per_example_gradient": _per_example_gradient_spec_runtime_path,
    }
    special_path = special_paths.get(operator.kind)

    if special_path is not None:
        return special_path(candidate)

    key = SPEC_PATH_KEYS.get(operator.kind)

    if key is None or key not in candidate.settings:
        return None

    value = candidate.settings[key]
    path_map = SPEC_PATH_TO_RUNTIME[operator.kind]
    path = path_map.get(value)

    if path is None:
        message = f"{key} value is not lowered by standard runtime: {value}"
        raise MaterializationError(message)

    return path


def _ggn_spec_runtime_path(candidate: Candidate) -> str | None:
    settings = candidate.settings
    jvp_key = SPEC_PATH_KEYS["ggnvp"]

    if (
        settings.get("ggn.loss_hessian_kernel") == "dense_global"
        and jvp_key not in settings
    ):
        return GGN_DENSE_PATH

    if jvp_key not in settings:
        return None

    value = settings[jvp_key]
    path_map = SPEC_PATH_TO_RUNTIME["ggnvp"]
    path = path_map.get(value)

    if path is None:
        message = f"{jvp_key} value is not lowered by standard runtime: {value}"
        raise MaterializationError(message)

    return path


def _fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    _require_fisher_expectation_path(candidate.settings)

    return _score_fisher_spec_runtime_path(
        candidate.settings,
        accumulation_key=SPEC_PATH_KEYS["fisher_vp"],
        score_path_key="fisher.score_grad_path",
        dense_path=FISHER_DENSE_PATH,
        block_path=FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        streaming_paths=FISHER_STREAMING_PATH_BY_SCORE_GRAD,
    )


def _require_fisher_expectation_path(settings: Mapping[str, Any]) -> None:
    expectation_key = "fisher.expectation_path"

    if expectation_key not in settings:
        message = "fisher.expectation_path is required for FisherVP rows"
        raise MaterializationError(message)

    if settings[expectation_key] != "explicit_full_expectation_score_rows":
        message = "fisher.expectation_path is unsupported"
        raise MaterializationError(message)


def _score_fisher_spec_runtime_path(
    settings: Mapping[str, Any],
    *,
    accumulation_key: str,
    score_path_key: str,
    dense_path: str,
    block_path: str,
    streaming_paths: Mapping[str, str],
) -> str | None:
    accumulation = settings.get(accumulation_key)

    if accumulation == "materialize_score_gradients":
        return _score_fisher_non_streaming_path(
            settings,
            score_path_key,
            "materialize_score_gradients",
            dense_path,
        )

    if accumulation == "blockwise_score_matrix":
        return _score_fisher_non_streaming_path(
            settings,
            score_path_key,
            "blockwise_score_matrix",
            block_path,
        )

    if accumulation not in {None, "streaming_dot_accumulate"}:
        message = f"{accumulation_key} value is not lowered: {accumulation}"
        raise MaterializationError(message)

    if accumulation is None:
        return None

    return _score_fisher_streaming_path(settings, score_path_key, streaming_paths)


def _score_fisher_non_streaming_path(
    settings: Mapping[str, Any],
    score_path_key: str,
    accumulation: str,
    path: str,
) -> str:
    if score_path_key in settings:
        message = f"{score_path_key} is not used with {accumulation}"
        raise MaterializationError(message)

    return path


def _score_fisher_streaming_path(
    settings: Mapping[str, Any],
    score_path_key: str,
    streaming_paths: Mapping[str, str],
) -> str:
    score_path = settings.get(score_path_key)

    if score_path_key not in settings:
        message = f"{score_path_key} is required for streaming rows"
        raise MaterializationError(message)

    path = streaming_paths.get(score_path) if isinstance(score_path, str) else None

    if path is not None:
        return path

    message = f"{score_path_key} value is not lowered by standard runtime: {score_path}"
    raise MaterializationError(message)


def _sampled_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    return _score_fisher_spec_runtime_path(
        candidate.settings,
        accumulation_key=SPEC_PATH_KEYS["sampled_fisher_vp"],
        score_path_key="sampled_fisher.score_grad_path",
        dense_path=SAMPLED_FISHER_DENSE_PATH,
        block_path=SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        streaming_paths=SAMPLED_FISHER_STREAMING_PATH_BY_SCORE_GRAD,
    )


def _empirical_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    grad_key = SPEC_PATH_KEYS["empirical_fisher_vp"]
    accumulation_key = "empirical_fisher.accumulation"
    grad_path = candidate.settings.get(grad_key)
    accumulation = candidate.settings.get(accumulation_key)

    if accumulation == "materialize_per_example_gradients":
        if grad_key in candidate.settings:
            message = (
                "empirical_fisher.grad_path is not used with "
                "materialize_per_example_gradients"
            )
            raise MaterializationError(message)

        return EMPIRICAL_FISHER_DENSE_PATH

    if accumulation == "blockwise_gradient_matrix":
        if grad_key in candidate.settings:
            message = (
                "empirical_fisher.grad_path is not used with blockwise_gradient_matrix"
            )
            raise MaterializationError(message)

        return EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH

    if accumulation not in {None, "streaming_dot_accumulate"}:
        message = f"empirical_fisher.accumulation value is not lowered: {accumulation}"
        raise MaterializationError(message)

    if grad_key not in candidate.settings:
        return None

    if not isinstance(grad_path, str):
        message = "empirical_fisher.grad_path must be a string"
        raise MaterializationError(message)

    path_map = SPEC_PATH_TO_RUNTIME["empirical_fisher_vp"]
    path = path_map.get(grad_path)

    if path is None:
        message = f"{grad_key} value is not lowered by standard runtime: {grad_path}"
        raise MaterializationError(message)

    return path


def _per_example_gradient_spec_runtime_path(candidate: Candidate) -> str | None:
    grad_key = SPEC_PATH_KEYS["per_example_gradient"]
    accumulation_key = "per_example_gradient.accumulation"
    grad_path = candidate.settings.get(grad_key)
    accumulation = candidate.settings.get(accumulation_key)

    if accumulation not in {"stacked_leading_axis", "blockwise_stacked"}:
        message = (
            f"per_example_gradient.accumulation value is not lowered: {accumulation}"
        )
        raise MaterializationError(message)

    if grad_key not in candidate.settings:
        return None

    if not isinstance(grad_path, str):
        message = "per_example_gradient.grad_path must be a string"
        raise MaterializationError(message)

    path_map = SPEC_PATH_TO_RUNTIME["per_example_gradient"]
    path = path_map.get(grad_path)

    if path is None:
        message = f"{grad_key} value is not lowered by standard runtime: {grad_path}"
        raise MaterializationError(message)

    return path


def _require_supported_standard_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    parameter_surface: ParameterSurface | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    activation_pack_hooks: ActivationPackHooks | None = None,
    activation_unpack_hooks: ActivationUnpackHooks | None = None,
    checkpoint_contexts: CheckpointContextFns | None = None,
) -> None:
    unsupported = tuple(
        key for key in candidate.settings if key not in SUPPORTED_STANDARD_SETTINGS
    )

    if unsupported:
        message = f"standard runtime settings are unsupported: {unsupported}"
        raise MaterializationError(message)

    path = _runtime_path(operator, candidate)
    _require_dtype_runtime_settings(candidate.settings)
    _require_teacher_output_settings(candidate.settings)
    _require_input_schedule_settings(operator, path, candidate.settings, batch_layout)
    _require_input_residency_settings(candidate.settings)
    _require_memory_residency_settings(
        operator,
        candidate.settings,
        mmap_residency,
    )
    _require_memory_recompute_settings(operator, candidate.settings)
    _require_output_buffer_settings(candidate.settings)
    _require_fusion_settings(candidate.settings, fusion_rewriter)
    _require_call_runtime_settings(candidate.settings)
    _require_stateful_module_path_settings(operator, path, candidate.settings)
    _require_gradient_graph_schedule_settings(operator, candidate.settings)
    _require_ggn_loss_hessian_settings(operator, candidate.settings)
    _require_ggn_batch_size_settings(operator, path, candidate.settings)
    _require_ggn_vjp_path_settings(operator, path, candidate.settings)
    _require_output_cotangent_block_settings(operator, path, candidate.settings)
    _require_lm_head_chunking_settings(candidate.settings, lm_head_chunker)
    _require_parameter_block_size_settings(
        operator,
        path,
        candidate.settings,
        parameter_surface,
    )
    _require_ggn_reuse_settings(operator, path, candidate.settings)
    _require_hvp_reuse_settings(operator, path, candidate.settings)
    _require_hvp_row_batch_size_settings(operator, path, candidate.settings)
    _require_vectorization_mode_settings(operator.kind, path, candidate.settings)
    _require_activation_runtime_settings(
        candidate.settings,
        activation_pack_hooks,
        activation_unpack_hooks,
        checkpoint_contexts,
    )
    _require_vectorization_setting_keys(operator.kind, path, candidate.settings)
    _require_transform_admission_settings(operator, path, candidate.settings)
    _require_gradient_value_reuse_settings(operator, path, candidate.settings)
    _require_jvp_linearize_reuse_settings(operator, path, candidate.settings)
    _require_vjp_closure_reuse_settings(operator, path, candidate.settings)
    _require_metric_runtime_settings(operator, path, candidate.settings)
    _require_inverse_metric_factor_reuse_settings(
        operator,
        path,
        candidate.settings,
    )
    _require_inverse_metric_multi_rhs_settings(operator, path, candidate.settings)
    _require_layout_runtime_settings(candidate.settings)


def _require_transform_admission_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    if _requires_torch_func_admission(operator, path, settings):
        try:
            admit_torch_func(settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error

    if path in {
        JVP_PATH,
        JVP_FORWARD_AD_PATH,
        HVP_JVP_GRAD_PATH,
        GGN_JVP_HESSIAN_VJP_PATH,
    }:
        try:
            admit_forward_ad(settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error


def _requires_torch_func_admission(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> bool:
    if path in {
        JVP_PATH,
        JVP_LINEARIZE_PATH,
        GRADIENT_TORCH_FUNC_PATH,
        GRADIENT_TORCH_FUNC_VALUE_PATH,
        VJP_PATH,
        HVP_JVP_GRAD_PATH,
        HVP_LINEARIZE_GRAD_PATH,
        GGN_JVP_HESSIAN_VJP_PATH,
        GGN_LINEARIZE_HESSIAN_VJP_PATH,
        FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        FISHER_SCORE_GRADIENT_VMAP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
        EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
        PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
        PER_EXAMPLE_GRADIENT_VMAP_PATH,
    }:
        return True

    if operator.kind == "ggnvp" and settings.get("ggn.vjp_path") == "torch_func_vjp":
        return True

    return settings.get("vectorization.mode") == "vmap"


def _require_vectorization_mode_settings(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    mode_key = "vectorization.mode"

    if mode_key not in settings:
        return

    mode = settings[mode_key]

    if mode == "manual_batch":
        if not _supports_vector_loop(operator_kind, path):
            message = (
                "vectorization.mode=manual_batch requires a supported vector-product "
                "path"
            )
            raise MaterializationError(message)

        _manual_vector_batch_size(settings)
        _require_vectorization_in_dims_setting(settings)

        return

    if mode == "vmap":
        if _supports_vector_vmap(operator_kind, path):
            _vmap_chunk_size(settings)
            _require_vectorization_in_dims_setting(settings)

            return

        if operator_kind == "hvp":
            message = "vectorization.mode=vmap requires linearize_grad HVP"
            raise MaterializationError(message)

        message = "vectorization.mode=vmap requires vector-axis vmap lowering"
        raise MaterializationError(message)

    if mode == "single_loop":
        if not _supports_vector_loop(operator_kind, path):
            message = (
                "vectorization.mode=single_loop requires a supported vector-product "
                "path"
            )
            raise MaterializationError(message)

        _require_vectorization_in_dims_setting(settings)

        return

    message = f"vectorization.mode is unsupported: {mode}"
    raise MaterializationError(message)


def _require_vectorization_setting_keys(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    mode = settings.get("vectorization.mode")

    if "vectorization.vmap_chunk_size" in settings:
        if mode != "vmap":
            message = "vectorization.vmap_chunk_size is only supported by vmap rows"
            raise MaterializationError(message)

        if not _supports_vector_vmap(
            operator_kind,
            path,
        ):
            message = "vectorization.vmap_chunk_size is only supported by vmap rows"
            raise MaterializationError(message)

    if "vectorization.batch_size" in settings and mode != "manual_batch":
        message = "vectorization.batch_size is only supported by manual_batch rows"
        raise MaterializationError(message)

    if "vectorization.in_dims" not in settings:
        return

    if _supports_vector_loop(operator_kind, path) and mode in {
        "single_loop",
        "manual_batch",
    }:
        return

    if _supports_vector_vmap(operator_kind, path) and mode == "vmap":
        return

    message = "vectorization.in_dims is unsupported for this operator path"
    raise MaterializationError(message)


def _supports_vector_loop(operator_kind: str, path: str | None) -> bool:
    return path in VECTOR_LOOP_RUNTIME_PATHS.get(operator_kind, ())


def _supports_vector_vmap(operator_kind: str, path: str | None) -> bool:
    return path in VECTOR_VMAP_RUNTIME_PATHS.get(operator_kind, ())


def _require_vectorization_in_dims_setting(settings: Mapping[str, Any]) -> None:
    if "vectorization.in_dims" in settings:
        return

    message = "vectorized rows require vectorization.in_dims"
    raise MaterializationError(message)


def _require_dtype_runtime_settings(settings: Mapping[str, Any]) -> None:
    model_compute = settings.get("dtype.model_compute")
    autodiff_compute = settings.get("dtype.autodiff_compute")

    if model_compute is None or autodiff_compute is None:
        return

    if model_compute == autodiff_compute:
        return

    if settings.get("call.path") == "stateful_module":
        return

    message = (
        "split model and autodiff compute dtypes require call.path=stateful_module"
    )
    raise MaterializationError(message)


def _require_teacher_output_settings(settings: Mapping[str, Any]) -> None:
    value = settings.get("teacher_outputs")

    if value is None:
        return

    if value in {"precomputed_cpu", "precomputed_cpu_pinned", "precomputed_gpu"}:
        return

    if value == "recomputed_with_equality_check":
        return

    message = f"teacher_outputs is unsupported: {value}"
    raise MaterializationError(message)


def _require_recomputed_teacher_objective(
    settings: Mapping[str, Any],
    teacher_objective: FunctionObjective | None,
) -> None:
    if settings.get("teacher_outputs") != "recomputed_with_equality_check":
        return

    if teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)


def _require_call_runtime_settings(settings: Mapping[str, Any]) -> None:
    if any(key in settings for key in FUNCTIONAL_CALL_FIELDS):
        try:
            admit_functional_call(settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error

    path = _require_call_path_settings(settings)
    _require_call_state_settings(settings, path)
    _require_call_return_settings(settings, path)


def _require_call_path_settings(settings: Mapping[str, Any]) -> Any:
    try:
        return admit_call_core_settings(settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error


def _require_call_state_settings(
    settings: Mapping[str, Any],
    path: Any,
) -> None:
    tied_weights = settings.get("call.tied_weights")

    if tied_weights not in {None, "preserve_alias_groups"}:
        message = f"call.tied_weights is unsupported: {tied_weights}"
        raise MaterializationError(message)

    parametrizations = settings.get("call.parametrizations")

    if parametrizations not in {None, "preserve_parametrizations"}:
        message = f"call.parametrizations is unsupported: {parametrizations}"
        raise MaterializationError(message)

    buffer_mutation = settings.get("call.buffer_mutation")

    if buffer_mutation == "declared_and_restored":
        if path == "stateful_module":
            _require_stateful_declared_restore_settings(settings)
        else:
            _require_declared_state_restore_settings(settings)
    elif buffer_mutation not in {None, "forbidden"}:
        message = f"call.buffer_mutation is unsupported: {buffer_mutation}"
        raise MaterializationError(message)


def _require_call_return_settings(
    settings: Mapping[str, Any],
    path: Any,
) -> None:
    return_type = settings.get("call.return_type")

    if return_type not in {None, "raw_tensor_tree"}:
        if path == "stateful_module" and (
            return_type == "model_output_object_with_declared_fields"
        ):
            return

        message = f"call.return_type is unsupported: {return_type}"
        raise MaterializationError(message)


def _require_declared_state_restore_settings(settings: Mapping[str, Any]) -> None:
    if (
        settings.get("call.path") != "functional_call"
        or settings.get("call.params") != "explicit_params"
        or settings.get("call.buffers") != "explicit_buffers"
    ):
        message = "declared state restoration requires explicit functional-call inputs"
        raise MaterializationError(message)

    try:
        admit_functional_call(settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error

    if settings["mutates_state"] is not True:
        message = "declared state restoration requires mutates_state=True"
        raise MaterializationError(message)


def _require_stateful_declared_restore_settings(settings: Mapping[str, Any]) -> None:
    try:
        admit_functional_call(settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error

    if settings["mutates_state"] is not True:
        message = "declared state restoration requires mutates_state=True"
        raise MaterializationError(message)


def _require_stateful_module_path_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if settings.get("call.path") != "stateful_module":
        return

    allowed_paths = {
        "gradient": (GRADIENT_PATH, GRADIENT_BACKWARD_MATERIALIZED_PATH),
        "vjp": (VJP_AUTOGRAD_OUTPUTS_PATH, VJP_BACKWARD_MATERIALIZED_PATH),
        "hvp": (HVP_REFERENCE_PATH,),
    }
    allowed = allowed_paths.get(operator.kind, ())

    if path in allowed:
        return

    message = "call.path=stateful_module requires a package-owned eager autograd path"
    raise MaterializationError(message)


def _require_stateful_module_execution(execution: StandardExecution) -> None:
    if execution.candidate.settings.get("call.path") != "stateful_module":
        return

    if execution.module is None:
        message = "call.path=stateful_module requires a module"
        raise MaterializationError(message)

    if execution.module_call is None:
        message = "call.path=stateful_module requires module_call"
        raise MaterializationError(message)

    _require_module_state_names(
        execution.module,
        execution.params,
        execution.buffers,
    )


def _require_module_state_names(
    module: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
) -> None:
    module_params = dict(module.named_parameters())
    module_buffers = dict(module.named_buffers())

    if set(params) != set(module_params):
        message = "module parameter names must match runtime params"
        raise MaterializationError(message)

    if set(buffers) != set(module_buffers):
        message = "module buffer names must match runtime buffers"
        raise MaterializationError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class _ModuleStateSlot:
    parent: torch.nn.Module
    name: str
    kind: str
    value: torch.Tensor | None


def _uses_stateful_module_call(execution: StandardExecution) -> bool:
    return execution.candidate.settings.get("call.path") == "stateful_module"


def _stateful_module_scalar_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], torch.Tensor]:
    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        output = _call_stateful_module(execution, active_params)

        if not isinstance(output, torch.Tensor) or output.ndim != 0:
            message = "stateful module scalar objective must return a scalar tensor"
            raise MaterializationError(message)

        return output

    return scalar_function


def _stateful_module_tensor_function(
    execution: StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    def tensor_function(active_params: ParameterTree) -> TensorTree:
        output = _call_stateful_module(execution, active_params)

        return _checked_function_output(
            execution.candidate.settings,
            output,
            "stateful module output",
        )

    return tensor_function


def _call_stateful_module(
    execution: StandardExecution,
    active_params: ParameterTree,
) -> object:
    if execution.module is None or execution.module_call is None:
        message = "stateful module execution requires module and module_call"
        raise MaterializationError(message)

    settings = execution.candidate.settings
    model_params = _stateful_model_tree(active_params, settings)
    model_buffers = _stateful_model_tree(execution.buffers, settings)
    model_batch = _stateful_model_batch(execution.batch, settings)
    slots = _replace_module_state(execution.module, model_params, model_buffers)

    try:
        if execution.compiled_model_forward is None:
            output = _invoke_stateful_module(
                execution.module,
                execution.module_call,
                model_batch,
            )
        else:
            output = execution.compiled_model_forward(model_batch)
    finally:
        _restore_module_state(slots)

    return _select_stateful_module_output(
        output,
        execution.module_call,
        execution.candidate.settings,
    )


def _stateful_model_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return _model_compute_tree(values, settings)


def _stateful_model_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    return _model_compute_batch(batch, settings)


def _call_compiled_model_forward(
    execution: StandardExecution,
    compiled_model_forward: Callable[[Batch], object],
    active_params: ParameterTree,
) -> object:
    if execution.module is None:
        message = "compile.boundary=model_forward requires module"
        raise MaterializationError(message)

    settings = execution.candidate.settings
    model_params = _stateful_model_tree(active_params, settings)
    model_buffers = _stateful_model_tree(execution.buffers, settings)
    model_batch = _stateful_model_batch(execution.batch, settings)
    slots = _replace_module_state(execution.module, model_params, model_buffers)

    try:
        return compiled_model_forward(model_batch)
    finally:
        _restore_module_state(slots)


def _model_compute_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    dtype = _dtype_setting(settings, "dtype.model_compute")

    return _runtime_named_tensor_dtype(values, dtype)


def _model_compute_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    dtype = _dtype_setting(settings, "dtype.model_compute")

    if dtype is None:
        return batch

    return {key: _runtime_batch_value(value, dtype) for key, value in batch.items()}


def _replace_module_state(
    module: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
) -> tuple[_ModuleStateSlot, ...]:
    slots = []

    for key, tensor in params.items():
        parent, name = _module_state_parent(module, key)
        parameters = parent.__dict__["_parameters"]
        slots.append(_ModuleStateSlot(parent, name, "parameter", parameters[name]))
        parameters[name] = tensor

    for key, tensor in buffers.items():
        parent, name = _module_state_parent(module, key)
        buffers_map = parent.__dict__["_buffers"]
        slots.append(_ModuleStateSlot(parent, name, "buffer", buffers_map[name]))
        buffers_map[name] = tensor

    return tuple(slots)


def _restore_module_state(slots: tuple[_ModuleStateSlot, ...]) -> None:
    for slot in reversed(slots):
        if slot.kind == "parameter":
            slot.parent.__dict__["_parameters"][slot.name] = slot.value
        elif slot.kind == "buffer":
            slot.parent.__dict__["_buffers"][slot.name] = slot.value
        else:
            message = f"module state slot kind is unsupported: {slot.kind}"
            raise MaterializationError(message)


def _module_state_parent(
    module: torch.nn.Module,
    key: str,
) -> tuple[torch.nn.Module, str]:
    parent_name, separator, state_name = key.rpartition(".")
    parent = module.get_submodule(parent_name) if separator else module

    return parent, state_name


def _invoke_stateful_module(
    module: torch.nn.Module,
    call: ModuleCallSpec,
    batch: Batch,
) -> object:
    args = tuple(_batch_value(batch, key) for key in call.positional_batch_keys)
    kwargs = {
        argument_name: _batch_value(batch, batch_key)
        for argument_name, batch_key in call.keyword_batch_keys.items()
    }

    return module(*args, **kwargs)


def _batch_value(batch: Batch, key: str) -> Any:
    if key in batch:
        return batch[key]

    message = f"module call batch field is missing: {key}"
    raise MaterializationError(message)


def _select_stateful_module_output(
    output: object,
    call: ModuleCallSpec,
    settings: Mapping[str, Any],
) -> object:
    return_type = settings.get("call.return_type")

    if return_type in {None, "raw_tensor_tree"}:
        if call.output_fields:
            message = "raw_tensor_tree module calls must not declare output fields"
            raise MaterializationError(message)

        return output

    if return_type != "model_output_object_with_declared_fields":
        message = f"call.return_type is unsupported: {return_type}"
        raise MaterializationError(message)

    if not call.output_fields:
        message = "model output object rows require declared output fields"
        raise MaterializationError(message)

    return {
        key: _select_stateful_module_output_path(output, path)
        for key, path in call.output_fields.items()
    }


def _select_stateful_module_output_path(
    output: object,
    path: tuple[str | int, ...],
) -> object:
    value = output

    for item in path:
        if isinstance(item, str):
            value = _select_named_output_field(value, item)
        else:
            value = _select_indexed_output_field(value, item)

    return value


def _select_named_output_field(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        for field_name, field_value in value.items():
            if field_name == key:
                return field_value

    if hasattr(value, key):
        return getattr(value, key)

    message = f"model output field is missing: {key}"
    raise MaterializationError(message)


def _select_indexed_output_field(value: object, index: int) -> object:
    if isinstance(value, (tuple, list)):
        try:
            return value[index]
        except IndexError as error:
            message = f"model output index is missing: {index}"
            raise MaterializationError(message) from error

    message = f"model output is not indexable at: {index}"
    raise MaterializationError(message)


def _require_input_schedule_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
    batch_layout_callback: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    per_example = settings.get("schedule.per_example")

    if per_example is None:
        if path in VMAP_RUNTIME_PATHS:
            message = "vmap_grad rows require schedule.per_example=vmap"
            raise MaterializationError(message)
    else:
        _require_batch_data_axis(operator, "schedule.per_example")
        _require_per_example_schedule(path, per_example)

    _require_per_example_batch_size_settings(path, settings)

    _require_per_token_schedule(operator, settings, batch_layout_callback)

    batch_layout = settings.get("input.batch_layout")

    if batch_layout not in {
        None,
        "dense_padded",
        "packed_with_inverse_permutation",
        "variable_length",
    }:
        message = f"input.batch_layout is unsupported: {batch_layout}"
        raise MaterializationError(message)

    if batch_layout not in {None, "dense_padded"} and batch_layout_callback is None:
        message = f"input.batch_layout requires input-layout binding: {batch_layout}"
        raise MaterializationError(message)

    length_grouping = settings.get("input.length_grouping")

    if length_grouping not in {None, "none", "exact_length_bucket"}:
        message = f"input.length_grouping is unsupported: {length_grouping}"
        raise MaterializationError(message)

    if length_grouping not in {None, "none"} and batch_layout_callback is None:
        message = (
            f"input.length_grouping requires input-order restoration: {length_grouping}"
        )
        raise MaterializationError(message)

    gradient_accumulation = settings.get("schedule.gradient_accumulation")

    if gradient_accumulation in {None, "single_step"}:
        if "batch.data_microbatch_size" in settings:
            message = (
                "batch.data_microbatch_size requires "
                "schedule.gradient_accumulation=microbatch_accumulate"
            )
            raise MaterializationError(message)

        return

    if gradient_accumulation != "microbatch_accumulate":
        message = (
            f"schedule.gradient_accumulation is unsupported: {gradient_accumulation}"
        )
        raise MaterializationError(message)

    _require_batch_data_axis(
        operator,
        "schedule.gradient_accumulation=microbatch_accumulate",
    )

    if (
        settings.get("compile.enabled") == "true"
        and settings.get("compile.boundary") == "loss_closure"
    ):
        message = "loss_closure compile boundary is incompatible with microbatching"
        raise MaterializationError(message)

    _data_microbatch_size(settings)


def _require_batch_data_axis(operator: OperatorSpec, label: str) -> None:
    if operator.data_axis == "batch":
        return

    message = f"{label} requires operator data_axis=batch"
    raise MaterializationError(message)


def _data_microbatch_size(settings: Mapping[str, Any]) -> int:
    key = "batch.data_microbatch_size"
    value = settings.get(key)

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _require_per_token_schedule(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    batch_layout_callback: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    token_block_size = _token_block_size(settings)
    schedule_value = settings.get("schedule.per_token")

    if schedule_value is None:
        if token_block_size is not None:
            message = "chunk.token_block_size requires schedule.per_token=loop"
            raise MaterializationError(message)

        return

    if schedule_value == "loop":
        if token_block_size is not None:
            _require_token_block_runtime(operator, settings)

        return

    if schedule_value == "packed":
        if settings.get("input.batch_layout") not in {
            "packed_with_inverse_permutation",
            "variable_length",
        }:
            message = "schedule.per_token=packed requires packed input binding"
            raise MaterializationError(message)

        if batch_layout_callback is None:
            message = "schedule.per_token=packed requires packed input binding"
            raise MaterializationError(message)

        return

    message = f"schedule.per_token is unsupported: {schedule_value}"
    raise MaterializationError(message)


def _token_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.token_block_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _require_token_block_runtime(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    if (
        operator.kind == "ggnvp"
        and settings.get("ggn.loss_hessian_path") == "closed_form_softmax_ce_kl"
    ):
        return

    message = "chunk.token_block_size requires closed-form CE/KL GGN"
    raise MaterializationError(message)


def _require_lm_head_chunking_settings(
    settings: Mapping[str, Any],
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    key = "chunk.lm_head_weight_chunk_bytes"
    value = settings.get(key)

    if value is None:
        return

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    if lm_head_chunker is None:
        message = f"{key} requires LM-head weight binding"
        raise MaterializationError(message)


def _require_per_example_schedule(path: str | None, value: Any) -> None:
    loop_paths = {
        FISHER_SCORE_GRADIENT_LOOP_PATH,
        FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        FISHER_BACKWARD_MATERIALIZED_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
        EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
    }

    if value == "loop":
        if path in loop_paths:
            return

        message = f"schedule.per_example=loop is incompatible with path: {path}"
        raise MaterializationError(message)

    if value == "vmap":
        if path in VMAP_RUNTIME_PATHS:
            return

        message = f"schedule.per_example=vmap is incompatible with path: {path}"
        raise MaterializationError(message)

    if value == "manual_batch":
        if path in {
            *FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
            *SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
            *EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
        }:
            return

        message = f"schedule.per_example=manual_batch is incompatible with path: {path}"
        raise MaterializationError(message)

    message = f"schedule.per_example is unsupported: {value}"
    raise MaterializationError(message)


def _require_per_example_batch_size_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    _require_per_example_batch_size_setting(
        path,
        settings,
        "batch.fisher_sample_batch_size",
        {
            FISHER_SCORE_GRADIENT_VMAP_PATH,
            SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
        },
        {
            *FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
            *SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS,
        },
    )
    _require_per_example_batch_size_setting(
        path,
        settings,
        "batch.empirical_example_batch_size",
        {EMPIRICAL_FISHER_GRADIENT_VMAP_PATH},
        set(EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS),
    )
    _require_per_example_gradient_block_size_setting(path, settings)


def _require_per_example_batch_size_setting(
    path: str | None,
    settings: Mapping[str, Any],
    key: str,
    allowed_paths: set[str],
    manual_paths: set[str],
) -> None:
    schedule = settings.get("schedule.per_example")

    if key not in settings:
        if schedule == "manual_batch" and path in manual_paths:
            message = f"{key} is required for schedule.per_example=manual_batch"
            raise MaterializationError(message)

        return

    if not (
        (schedule == "vmap" and path in allowed_paths)
        or (schedule == "manual_batch" and path in manual_paths)
    ):
        message = (
            f"{key} requires schedule.per_example=vmap or manual_batch on a "
            "matching path"
        )
        raise MaterializationError(message)

    value = settings[key]

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)


def _require_per_example_gradient_block_size_setting(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    key = "batch.per_example_block_size"
    accumulation = settings.get("per_example_gradient.accumulation")
    per_example_paths = {
        PER_EXAMPLE_GRADIENT_LOOP_PATH,
        PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
        PER_EXAMPLE_GRADIENT_BACKWARD_PATH,
        PER_EXAMPLE_GRADIENT_VMAP_PATH,
    }

    if key not in settings:
        if accumulation == "blockwise_stacked" and path in per_example_paths:
            message = f"{key} is required for blockwise_stacked"
            raise MaterializationError(message)

        return

    if accumulation != "blockwise_stacked" or path not in per_example_paths:
        message = f"{key} requires per_example_gradient.accumulation=blockwise_stacked"
        raise MaterializationError(message)

    value = settings[key]

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)


def _require_input_residency_settings(settings: Mapping[str, Any]) -> None:
    residency = settings.get("input.residency")
    host_to_device = settings.get("input.host_to_device")

    if residency is None and host_to_device is None:
        return

    if residency not in {"cpu_staged", "cpu_pinned", "gpu"}:
        message = f"input.residency is unsupported: {residency}"
        raise MaterializationError(message)

    if host_to_device not in {"outside_measured_call", "inside_measured_call"}:
        message = f"input.host_to_device is unsupported: {host_to_device}"
        raise MaterializationError(message)


def _require_memory_residency_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> None:
    vector_residency = settings.get("memory.vector_residency")
    factor_residency = settings.get("memory.factor_residency")
    intermediate_residency = settings.get("memory.intermediate_residency")

    if vector_residency is not None:
        _require_runtime_residency(
            vector_residency,
            "memory.vector_residency",
            allow_mmap=mmap_residency is not None,
        )

    if factor_residency is not None:
        _require_runtime_residency(
            factor_residency,
            "memory.factor_residency",
            allow_mmap=mmap_residency is not None,
        )

    if intermediate_residency is None:
        return

    if operator.kind in {"composition", "ggnvp"}:
        _require_runtime_residency(
            intermediate_residency,
            "memory.intermediate_residency",
            allow_mmap=False,
        )

        if (
            operator.kind == "composition"
            and settings.get("composition.execution") == "fuse_adjacent_children"
        ):
            message = (
                "memory.intermediate_residency requires visible composition child "
                "boundaries"
            )
            raise MaterializationError(message)

        return

    if intermediate_residency in {"gpu", "cpu_staged", "cpu_pinned"}:
        message = "memory.intermediate_residency requires named intermediate boundaries"
        raise MaterializationError(message)

    message = f"memory.intermediate_residency is unsupported: {intermediate_residency}"
    raise MaterializationError(message)


def _require_memory_recompute_settings(
    _: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    for key in (
        "memory.primal_outputs",
        "memory.jvp_outputs",
        "memory.output_cotangents",
    ):
        value = settings.get(key)

        if value is None or value == "retain":
            continue

        if value == "recompute":
            message = (
                f"{key}=recompute requires package-owned output recompute lowering"
            )
            raise MaterializationError(message)

        message = f"{key} is unsupported: {value}"
        raise MaterializationError(message)


def _require_gradient_graph_schedule_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    graph_schedule = settings.get("gradient.graph_schedule")

    if graph_schedule is None:
        return

    if operator.kind != "gradient":
        message = "gradient.graph_schedule applies only to gradient rows"
        raise MaterializationError(message)

    if graph_schedule not in {"build_once", "rebuild_per_call"}:
        message = f"gradient.graph_schedule is unsupported: {graph_schedule}"
        raise MaterializationError(message)


def _require_runtime_residency(
    value: Any,
    key: str,
    *,
    allow_mmap: bool,
) -> None:
    if value in {"cpu_staged", "cpu_pinned", "gpu"}:
        return

    if value == "mmap_cpu":
        if allow_mmap:
            return

        message = f"{key}=mmap_cpu requires memory-mapped tensor metadata"
        raise MaterializationError(message)

    message = f"{key} is unsupported: {value}"
    raise MaterializationError(message)


def _require_output_buffer_settings(settings: Mapping[str, Any]) -> None:
    value = settings.get("memory.output_buffers")

    if value is None or value == "fresh_allocation":
        return

    if value == "preallocated":
        return

    message = f"memory.output_buffers is unsupported: {value}"
    raise MaterializationError(message)


def _require_fusion_settings(
    settings: Mapping[str, Any],
    fusion_rewriter: Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None,
) -> None:
    for key, fused_values in _fusion_value_domains().items():
        value = settings.get(key)

        if value is None or value == "model_default":
            continue

        if value in fused_values:
            if fusion_rewriter is None:
                message = f"{key}={value} requires a registered fused implementation"
                raise MaterializationError(message)

            continue

        message = f"{key} is unsupported: {value}"
        raise MaterializationError(message)


def _fusion_value_domains() -> Mapping[str, set[str]]:
    return {
        "fusion.norm": {"fused_rmsnorm", "fused_layernorm"},
        "fusion.mlp": {"fused_mlp"},
        "fusion.rope": {"fused_rope"},
        "fusion.logits": {"fused_logits_projection"},
        "fusion.loss": {"fused_ce", "fused_kl"},
    }


def _runtime_fusion_module(
    module: torch.nn.Module | None,
    candidate: Candidate,
    fusion_rewriter: Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None,
) -> torch.nn.Module | None:
    if not _has_fused_setting(candidate.settings):
        return module

    if module is None:
        message = "fused rows require a module"
        raise MaterializationError(message)

    if fusion_rewriter is None:
        message = "fused rows require a registered fused implementation"
        raise MaterializationError(message)

    return fusion_rewriter(module, candidate)


def _has_fused_setting(settings: Mapping[str, Any]) -> bool:
    for key, fused_values in _fusion_value_domains().items():
        if settings.get(key) in fused_values:
            return True

    return False


def _require_activation_runtime_settings(
    settings: Mapping[str, Any],
    activation_pack_hooks: ActivationPackHooks | None,
    activation_unpack_hooks: ActivationUnpackHooks | None,
    checkpoint_contexts: CheckpointContextFns | None,
) -> None:
    if not _has_activation_settings(settings):
        return

    recompute = settings.get("activation.recompute")
    offload = settings.get("activation.offload")

    if recompute not in {
        "none",
        "checkpoint_non_reentrant_by_layer",
        "checkpoint_selective",
        "manual_recompute",
    }:
        message = f"activation.recompute is unsupported: {recompute}"
        raise MaterializationError(message)

    if offload not in {"none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}:
        message = f"activation.offload is unsupported: {offload}"
        raise MaterializationError(message)

    if (
        recompute == "checkpoint_selective"
        and settings.get("checkpoint.context_fn") != "declared_context_pair"
    ):
        message = (
            "checkpoint_selective requires checkpoint.context_fn=declared_context_pair"
        )
        raise MaterializationError(message)

    if recompute == "checkpoint_non_reentrant_by_layer":
        message = (
            "checkpoint_non_reentrant_by_layer requires package-owned layer "
            "checkpoint lowering"
        )
        raise MaterializationError(message)

    if recompute in {"checkpoint_non_reentrant_by_layer", "checkpoint_selective"}:
        _admit_checkpoint_runtime(settings)
        _require_checkpoint_context_binding(settings, checkpoint_contexts)

        if offload == "custom_saved_tensor_hooks":
            _require_activation_hook_binding(
                settings,
                activation_pack_hooks,
                activation_unpack_hooks,
            )

        return

    _require_disabled_checkpoint_settings(settings, recompute)

    if offload == "custom_saved_tensor_hooks":
        _require_activation_hook_binding(
            settings,
            activation_pack_hooks,
            activation_unpack_hooks,
        )


def _admit_checkpoint_runtime(settings: Mapping[str, Any]) -> None:
    try:
        admit_checkpoint(settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error


def _require_checkpoint_context_binding(
    settings: Mapping[str, Any],
    checkpoint_contexts: CheckpointContextFns | None,
) -> None:
    if settings.get("checkpoint.context_fn") != "declared_context_pair":
        return

    context_id = settings.get("checkpoint.context_fn_callable")

    if not isinstance(context_id, str):
        message = "checkpoint.context_fn=declared_context_pair requires context id"
        raise MaterializationError(message)

    if checkpoint_contexts is None or context_id not in checkpoint_contexts:
        message = f"checkpoint context is not registered: {context_id}"
        raise MaterializationError(message)


def _require_activation_hook_binding(
    settings: Mapping[str, Any],
    activation_pack_hooks: ActivationPackHooks | None,
    activation_unpack_hooks: ActivationUnpackHooks | None,
) -> None:
    pack_hook_id = settings.get("activation.pack_hook")
    unpack_hook_id = settings.get("activation.unpack_hook")

    if not isinstance(pack_hook_id, str) or not isinstance(unpack_hook_id, str):
        message = "custom_saved_tensor_hooks requires activation pack and unpack hooks"
        raise MaterializationError(message)

    if activation_pack_hooks is None or pack_hook_id not in activation_pack_hooks:
        message = f"activation pack hook is not registered: {pack_hook_id}"
        raise MaterializationError(message)

    if activation_unpack_hooks is None or unpack_hook_id not in activation_unpack_hooks:
        message = f"activation unpack hook is not registered: {unpack_hook_id}"
        raise MaterializationError(message)


def _require_disabled_checkpoint_settings(
    settings: Mapping[str, Any],
    recompute: str,
) -> None:
    disabled = {
        "checkpoint.use_reentrant": "false",
        "checkpoint.early_stop": "false",
        "checkpoint.preserve_rng_state": "false",
        "checkpoint.determinism_check": "none",
        "checkpoint.context_fn": "none",
        "checkpoint.moves_to_new_device": "false",
        "checkpoint.uses_global_state": "false",
    }

    for key, value in disabled.items():
        if key in settings and settings[key] != value:
            message = f"activation.recompute={recompute} requires {key}={value}"
            raise MaterializationError(message)


def _require_gradient_value_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="gradient",
        setting_key="gradient.value_reuse",
        default_value="gradient_only",
        required_value="gradient_and_primal_value",
        required_path=GRADIENT_TORCH_FUNC_VALUE_PATH,
        path_message="gradient_and_primal_value requires torch_func_grad_and_value",
    )


def _require_jvp_linearize_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="jvp",
        setting_key="jvp.linearize_reuse",
        default_value="none",
        required_value="reuse_at_same_primal",
        required_path=JVP_LINEARIZE_PATH,
        path_message="reuse_at_same_primal requires torch_func_linearize",
    )


def _require_vjp_closure_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="vjp",
        setting_key="vjp.closure_reuse",
        default_value="none",
        required_value="reuse_vjp_closure_at_same_primal",
        required_path=VJP_PATH,
        path_message="reuse_vjp_closure_at_same_primal requires torch_func_vjp",
    )


def _require_path_coupled_reuse_setting(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
    *,
    operator_kind: str,
    setting_key: str,
    default_value: str,
    required_value: str,
    required_path: str,
    path_message: str,
) -> None:
    if operator.kind != operator_kind:
        return

    value = settings.get(setting_key)

    if value is None or value == default_value:
        return

    if value != required_value:
        message = f"{setting_key} is unsupported: {value}"
        raise MaterializationError(message)

    if path != required_path:
        raise MaterializationError(path_message)


def _require_metric_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if operator.kind in {"sqrt_metric", "inverse_sqrt_metric"}:
        _require_sqrt_metric_operator_settings(operator, path, settings)
        return

    if operator.kind == "metric":
        _require_metric_block_schedule(operator, settings, "metric.block_schedule")
        _require_metric_accumulation_settings(path, settings)
        return

    if operator.kind == "metric_inner":
        _require_metric_inner_runtime_settings(operator, path, settings)
        return

    if operator.kind == "inverse_metric":
        _require_inverse_metric_operator_settings(operator, path, settings)
        return

    if operator.kind == "inverse_metric_inner":
        _require_inverse_metric_inner_runtime_settings(operator, path, settings)
        return

    _reject_stray_metric_runtime_settings(settings)


def _require_sqrt_metric_operator_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    _require_sqrt_metric_runtime_settings(path, settings)

    if path == SQRT_METRIC_LANCZOS_PATH:
        _require_metric_representation(operator, ("matrix_free",))
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)

        if "metric.block_schedule" in settings:
            message = "metric.block_schedule does not apply to Lanczos sqrt rows"
            raise MaterializationError(message)

        if "inverse_metric.block_schedule" in settings:
            message = (
                "inverse_metric.block_schedule does not apply to Lanczos sqrt rows"
            )
            raise MaterializationError(message)

        if _has_inverse_metric_settings(settings):
            message = "inverse metric settings do not apply to sqrt rows"
            raise MaterializationError(message)

        return

    if _has_metric_runtime_settings(settings) or _has_inverse_metric_settings(settings):
        message = "metric solve and multiply settings do not apply to sqrt rows"
        raise MaterializationError(message)


def _require_metric_inner_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if path == METRIC_INNER_MULTIPLY_REDUCE_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_block_schedule(operator, settings, "metric.block_schedule")
        _require_metric_accumulation_settings(metric_path, settings)
        return

    if path == METRIC_INNER_SQRT_REDUCE_PATH:
        sqrt_path = _sqrt_metric_runtime_path_from_settings(settings)
        _require_sqrt_metric_runtime_settings(sqrt_path, settings)

        if _has_metric_multiply_settings(settings):
            message = "metric multiply settings apply only to multiply_then_reduce"
            raise MaterializationError(message)

        return

    if _has_metric_multiply_settings(settings) or _has_sqrt_metric_settings(settings):
        message = "metric multiply settings apply only to multiply_then_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_operator_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_inverse_metric_tolerance_path(operator, path)
    _require_inverse_metric_iteration_budget_settings(path, settings)
    _require_metric_block_schedule(
        operator,
        settings,
        "inverse_metric.block_schedule",
    )

    if path == INVERSE_METRIC_CG_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)
        return

    if _has_metric_multiply_settings(settings):
        message = (
            "metric settings apply only to metric rows and conjugate_gradient "
            "inverse_metric rows"
        )
        raise MaterializationError(message)


def _require_inverse_metric_inner_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_inverse_metric_inner_tolerance_reduction(operator, path)

    if path == INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        _require_inverse_metric_inner_solve_settings(operator, settings)
        return

    if path == INVERSE_METRIC_INNER_SQRT_REDUCE_PATH:
        sqrt_path = _sqrt_metric_runtime_path_from_settings(settings)
        _require_sqrt_metric_runtime_settings(sqrt_path, settings)

        if _has_inverse_metric_settings(settings):
            message = "inverse metric solve settings apply only to solve_then_reduce"
            raise MaterializationError(message)

        return

    if _has_inverse_metric_settings(settings):
        message = "inverse metric solve settings apply only to solve_then_reduce"
        raise MaterializationError(message)

    if _has_sqrt_metric_settings(settings):
        message = "sqrt metric settings apply only to sqrt_apply_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_inner_solve_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    inverse_path = _inverse_metric_runtime_path_from_settings(settings)
    _require_inverse_metric_inner_tolerance_path(operator, inverse_path)
    _require_inverse_metric_iteration_budget_settings(inverse_path, settings)
    _require_metric_block_schedule(
        operator,
        settings,
        "inverse_metric.block_schedule",
    )

    if inverse_path == INVERSE_METRIC_CG_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)
    elif _has_metric_multiply_settings(settings):
        message = (
            "metric multiply settings apply only to conjugate_gradient "
            "solve_then_reduce"
        )
        raise MaterializationError(message)

    if _has_sqrt_metric_settings(settings):
        message = "sqrt metric settings apply only to sqrt_apply_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_tolerance_path(
    operator: OperatorSpec,
    path: str,
) -> None:
    if _inverse_metric_tolerance(operator) is None:
        return

    if path == INVERSE_METRIC_CG_PATH:
        return

    message = "inverse metric tol requires conjugate_gradient"
    raise MaterializationError(message)


def _require_inverse_metric_inner_tolerance_reduction(
    operator: OperatorSpec,
    path: str,
) -> None:
    if _inverse_metric_tolerance(operator) is None:
        return

    if path == INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        return

    message = "inverse_metric_inner tol requires solve_then_reduce"
    raise MaterializationError(message)


def _require_inverse_metric_inner_tolerance_path(
    operator: OperatorSpec,
    path: str,
) -> None:
    if _inverse_metric_tolerance(operator) is None:
        return

    if path == INVERSE_METRIC_CG_PATH:
        return

    message = "inverse_metric_inner tol requires conjugate_gradient"
    raise MaterializationError(message)


def _reject_stray_metric_runtime_settings(settings: Mapping[str, Any]) -> None:
    if not (
        _has_metric_runtime_settings(settings)
        or _has_inverse_metric_settings(settings)
        or _has_sqrt_metric_settings(settings)
    ):
        return

    message = (
        "metric settings apply only to metric rows and conjugate_gradient "
        "inverse_metric rows"
    )
    raise MaterializationError(message)


def _has_metric_multiply_settings(settings: Mapping[str, Any]) -> bool:
    return "metric.multiply_path" in settings or "metric.accumulation" in settings


def _has_metric_runtime_settings(settings: Mapping[str, Any]) -> bool:
    return (
        _has_metric_multiply_settings(settings)
        or "metric.block_schedule" in settings
        or "inverse_metric.block_schedule" in settings
    )


def _has_inverse_metric_settings(settings: Mapping[str, Any]) -> bool:
    return (
        "inverse_metric.solve_path" in settings
        or "inverse_metric.iteration_budget" in settings
        or "inverse_metric.preconditioner" in settings
        or "inverse_metric.block_schedule" in settings
    )


def _has_sqrt_metric_settings(settings: Mapping[str, Any]) -> bool:
    return (
        "sqrt_metric.factor_path" in settings
        or "sqrt_metric.lanczos_iterations" in settings
    )


def _require_sqrt_metric_runtime_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    if path == SQRT_METRIC_LANCZOS_PATH:
        _sqrt_metric_lanczos_iterations(settings)

        return

    if "sqrt_metric.lanczos_iterations" in settings:
        message = "sqrt_metric.lanczos_iterations requires matrix_free_lanczos"
        raise MaterializationError(message)

    _sqrt_metric_runtime_path_from_settings(settings)


def _require_inverse_metric_factor_reuse_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    value = settings.get("inverse_metric.factor_reuse")

    if value is None:
        return

    if operator.kind != "inverse_metric":
        message = "inverse_metric.factor_reuse applies only to inverse_metric rows"
        raise MaterializationError(message)

    if value == "refactor_each_rhs":
        return

    if value == "reuse_factor_across_rhs":
        if settings.get("vectorization.mode") not in {"single_loop", "manual_batch"}:
            message = "reuse_factor_across_rhs requires vectorized inverse metric input"
            raise MaterializationError(message)

        if path not in INVERSE_METRIC_FACTOR_REUSE_PATHS:
            message = "reuse_factor_across_rhs requires a factor-reuse solve path"
            raise MaterializationError(message)

        return

    message = f"inverse_metric.factor_reuse is unsupported: {value}"
    raise MaterializationError(message)


def _require_inverse_metric_multi_rhs_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    value = settings.get("inverse_metric.multi_rhs")
    mode = settings.get("vectorization.mode")

    if value is None:
        if operator.kind == "inverse_metric" and mode in {
            "single_loop",
            "manual_batch",
        }:
            message = "vectorized inverse_metric rows require inverse_metric.multi_rhs"
            raise MaterializationError(message)

        return

    if operator.kind != "inverse_metric":
        message = "inverse_metric.multi_rhs applies only to inverse_metric rows"
        raise MaterializationError(message)

    if value == "single_column":
        return

    if value != "block":
        message = f"inverse_metric.multi_rhs is unsupported: {value}"
        raise MaterializationError(message)

    if mode not in {"single_loop", "manual_batch"}:
        message = "inverse_metric.multi_rhs=block requires stacked vectors"
        raise MaterializationError(message)

    if path not in INVERSE_METRIC_FACTOR_REUSE_PATHS:
        message = "inverse_metric.multi_rhs=block requires a batched solve path"
        raise MaterializationError(message)


def _require_inverse_metric_iteration_budget_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    if "inverse_metric.iteration_budget" not in settings:
        return

    if path != INVERSE_METRIC_CG_PATH:
        message = "inverse_metric.iteration_budget applies only to conjugate_gradient"
        raise MaterializationError(message)

    _inverse_metric_iteration_budget(settings)


def _require_metric_block_schedule(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    axis_key: str,
) -> None:
    value = settings.get(axis_key)

    if value is None:
        return

    representation = _metric_representation(operator)
    kind = _metric_representation_kind(operator)

    if kind not in {"block_diagonal", "kfac_factors"}:
        message = f"{axis_key} requires blocks or KFAC factors"
        raise MaterializationError(message)

    schedule = representation.get("block_schedule")

    if not isinstance(schedule, str):
        message = f"{axis_key} requires representation.block_schedule"
        raise MaterializationError(message)

    if value != schedule:
        message = f"{axis_key} must match representation.block_schedule"
        raise MaterializationError(message)


def _require_layout_runtime_settings(settings: Mapping[str, Any]) -> None:
    flatten_order = settings.get("layout.flatten_order")

    if flatten_order is not None and flatten_order != "canonical_parameter_order":
        message = f"layout.flatten_order is unsupported: {flatten_order}"
        raise MaterializationError(message)

    _layout_tree_input(settings, "layout.params")
    _layout_tree_input(settings, "layout.vector")
    _layout_output(settings)
    _layout_single_value(
        settings,
        "layout.aliasing",
        "preserve_tied_weight_aliases",
    )
    _layout_single_value(
        settings,
        "layout.parametrizations",
        "preserve_active_parametrizations",
    )
    _layout_vector_ops(settings)


def _layout_tree_input(settings: Mapping[str, Any], key: str) -> None:
    value = settings.get(key)

    if value is None or value == "parameter_tree":
        return

    if value in {"flat_contiguous", "per_layer_flat", "per_block_flat"}:
        return

    message = f"{key}={value} requires tree reconstruction support"
    raise MaterializationError(message)


def _layout_output(settings: Mapping[str, Any]) -> str:
    value = settings.get("layout.output")

    if value is None or value == "parameter_tree":
        return "parameter_tree"

    if value == "flat_contiguous":
        return "flat_contiguous"

    if value in {"per_layer_flat", "per_block_flat"}:
        return "parameter_tree"

    message = f"layout.output is unsupported: {value}"
    raise MaterializationError(message)


def _layout_single_value(
    settings: Mapping[str, Any],
    key: str,
    expected: str,
) -> None:
    value = settings.get(key)

    if value is None or value == expected:
        return

    message = f"{key} is unsupported: {value}"
    raise MaterializationError(message)


def _layout_vector_ops(settings: Mapping[str, Any]) -> str:
    value = settings.get("layout.vector_ops")

    if value is None or value == "python_loop":
        return "python_loop"

    if value == "foreach":
        return "foreach"

    message = f"layout.vector_ops is unsupported: {value}"
    raise MaterializationError(message)


def _tree_dot_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    left = _runtime_intermediate_tree(left, settings)
    right = _runtime_intermediate_tree(right, settings)
    left = _accumulation_tree(left, settings)
    right = _accumulation_tree(right, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_dot_foreach(left, right)

    return tree_dot(left, right)


def _tree_add_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    left = _runtime_intermediate_tree(left, settings)
    right = _runtime_intermediate_tree(right, settings)
    left = _accumulation_tree(left, settings)
    right = _accumulation_tree(right, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_add_foreach(left, right)

    return tree_map2(torch.add, left, right)


def _dot_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = _runtime_intermediate_tensor(left, settings)
    right = _runtime_intermediate_tensor(right, settings)

    return torch.dot(
        _accumulation_tensor(left, settings),
        _accumulation_tensor(right, settings),
    )


def _batched_dot_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = _runtime_intermediate_tensor(left, settings)
    right = _runtime_intermediate_tensor(right, settings)
    left_accumulation = _accumulation_tensor(left, settings)
    right_accumulation = _accumulation_tensor(right, settings)

    return torch.sum(left_accumulation * right_accumulation, dim=1)


def _matmul_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = _runtime_intermediate_tensor(left, settings)
    right = _runtime_intermediate_tensor(right, settings)

    return _accumulation_tensor(left, settings) @ _accumulation_tensor(
        right,
        settings,
    )


def _tree_scale_runtime(
    settings: Mapping[str, Any],
    tree: TensorTree,
    scale: float,
) -> TensorTree:
    tree = _runtime_intermediate_tree(tree, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_mul_foreach(tree, scale)

    return tree_map(lambda tensor: tensor * scale, tree)


def _tree_elementwise_mul_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    left = _runtime_intermediate_tree(left, settings)
    right = _runtime_intermediate_tree(right, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_elementwise_mul_foreach(left, right)

    return tree_map2(torch.mul, left, right)


def _tree_elementwise_div_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    left = _runtime_intermediate_tree(left, settings)
    right = _runtime_intermediate_tree(right, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_elementwise_div_foreach(left, right)

    return tree_map2(torch.div, left, right)


def _tree_add_scalar_runtime(
    settings: Mapping[str, Any],
    tree: TensorTree,
    scalar: float,
) -> TensorTree:
    tree = _runtime_intermediate_tree(tree, settings)

    if _layout_vector_ops(settings) == "foreach":
        return tree_add_scalar_foreach(tree, scalar)

    return tree_map(lambda tensor: tensor + scalar, tree)


def _runtime_intermediate_tree(
    tree: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return tree_map(lambda tensor: _runtime_intermediate_tensor(tensor, settings), tree)


def _runtime_intermediate_residency_tree(
    tree: TensorTree,
    settings: Mapping[str, Any],
    intermediate_transform: IntermediateTransform | None = None,
) -> TensorTree:
    residency = settings.get("memory.intermediate_residency")
    result = tree

    if residency is not None:
        result = _tree_residency(tree, residency, "memory.intermediate_residency")

    if intermediate_transform is None:
        return result

    return intermediate_transform(result)


def _runtime_intermediate_tensor(
    tensor: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    dtype = _dtype_setting(settings, "dtype.intermediate")

    if dtype is None or not tensor.is_floating_point():
        return tensor

    return tensor.to(dtype=dtype)


def _require_ggn_vjp_path_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if operator.kind != "ggnvp":
        return

    value = settings.get("ggn.vjp_path")

    if path == GGN_DENSE_PATH:
        if value is not None:
            message = "ggn.vjp_path is not used with dense_global"
            raise MaterializationError(message)

        return

    if value is None:
        message = "ggn.vjp_path is required for JVP-Hessian-VJP rows"
        raise MaterializationError(message)

    if value not in {
        "torch_func_vjp",
        "autograd_grad_outputs",
    }:
        message = f"ggn.vjp_path is unsupported: {value}"
        raise MaterializationError(message)


def _require_ggn_loss_hessian_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    if operator.kind != "ggnvp":
        return

    path = settings.get("ggn.loss_hessian_path")
    kernel = settings.get("ggn.loss_hessian_kernel")
    chunk_key = "chunk.class_block_size_with_exact_global_normalization"

    if path is None and kernel is None:
        if chunk_key in settings:
            message = f"{chunk_key} requires two_pass_chunked_global"
            raise MaterializationError(message)

        return

    if path is None or kernel is None:
        message = "GGN loss-Hessian rows require path and kernel settings"
        raise MaterializationError(message)

    if path == "closed_form_softmax_ce_kl":
        if kernel not in {
            "dense_global",
            "streaming_global",
            "two_pass_chunked_global",
        }:
            message = f"ggn.loss_hessian_kernel is unsupported: {kernel}"
            raise MaterializationError(message)

        if kernel == "two_pass_chunked_global":
            _class_block_size_with_exact_global_normalization(settings)
        elif chunk_key in settings:
            message = f"{chunk_key} requires two_pass_chunked_global"
            raise MaterializationError(message)

        return

    if path != "autodiff_loss_hvp":
        message = f"ggn.loss_hessian_path is not lowered by standard runtime: {path}"
        raise MaterializationError(message)

    if kernel != "dense_global":
        message = (
            f"ggn.loss_hessian_kernel is not lowered by standard runtime: {kernel}"
        )
        raise MaterializationError(message)

    if chunk_key in settings:
        message = f"{chunk_key} requires two_pass_chunked_global"
        raise MaterializationError(message)


def _require_ggn_batch_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    batch_size = _ggn_batch_size(settings)

    if batch_size is None:
        return

    if operator.kind != "ggnvp":
        message = "batch.ggn_batch_size applies only to GGNVP rows"
        raise MaterializationError(message)

    if path != GGN_DENSE_PATH:
        message = "batch.ggn_batch_size requires dense GGN"
        raise MaterializationError(message)


def _ggn_batch_size(settings: Mapping[str, Any]) -> int | None:
    key = "batch.ggn_batch_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _hvp_row_batch_size(settings: Mapping[str, Any]) -> int | None:
    key = "batch.hvp_row_batch_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _require_hvp_row_batch_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    batch_size = _hvp_row_batch_size(settings)

    if batch_size is None:
        return

    if operator.kind != "hvp":
        message = "batch.hvp_row_batch_size applies only to HVP rows"
        raise MaterializationError(message)

    if path != HVP_REFERENCE_PATH:
        message = "batch.hvp_row_batch_size requires reverse_over_reverse"
        raise MaterializationError(message)


def _require_hvp_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    graph_schedule = settings.get("hvp.graph_schedule")
    primal_reuse = settings.get("hvp.primal_reuse")
    gradient_reuse = settings.get("hvp.gradient_reuse")

    if operator.kind != "hvp":
        if graph_schedule is not None:
            message = "hvp.graph_schedule applies only to HVP rows"
            raise MaterializationError(message)

        if primal_reuse is not None:
            message = "hvp.primal_reuse applies only to HVP rows"
            raise MaterializationError(message)

        if gradient_reuse is not None:
            message = "hvp.gradient_reuse applies only to HVP rows"
            raise MaterializationError(message)

        return

    if graph_schedule == "retain_graph_across_vectors":
        _require_hvp_reverse_reuse_settings(path, settings)

        if primal_reuse != "reuse_primal":
            message = (
                "retain_graph_across_vectors requires hvp.primal_reuse=reuse_primal"
            )
            raise MaterializationError(message)
    elif graph_schedule not in {None, "rebuild_graph_per_vector"}:
        message = f"hvp.graph_schedule is unsupported: {graph_schedule}"
        raise MaterializationError(message)

    if primal_reuse == "reuse_primal":
        _require_hvp_reverse_reuse_settings(path, settings)
    elif primal_reuse not in {None, "recompute_primal"}:
        message = f"hvp.primal_reuse is unsupported: {primal_reuse}"
        raise MaterializationError(message)

    if gradient_reuse in {None, "recompute_gradient"}:
        return

    if gradient_reuse != "reuse_gradient_closure":
        message = f"hvp.gradient_reuse is unsupported: {gradient_reuse}"
        raise MaterializationError(message)

    if path != HVP_LINEARIZE_GRAD_PATH:
        message = "reuse_gradient_closure requires linearize_grad"
        raise MaterializationError(message)


def _require_hvp_reverse_reuse_settings(
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if path != HVP_REFERENCE_PATH:
        message = "HVP graph and primal reuse require reverse_over_reverse"
        raise MaterializationError(message)

    if settings.get("vectorization.mode") != "single_loop":
        message = "HVP graph and primal reuse require vectorization.mode=single_loop"
        raise MaterializationError(message)

    if "vectorization.in_dims" not in settings:
        message = "HVP graph and primal reuse require vectorization.in_dims"
        raise MaterializationError(message)


def _require_ggn_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    jvp_reuse = settings.get("ggn.jvp_reuse")
    cotangent_reuse = settings.get("ggn.cotangent_reuse")

    if operator.kind != "ggnvp":
        if jvp_reuse is not None:
            message = "ggn.jvp_reuse applies only to GGNVP rows"
            raise MaterializationError(message)

        if cotangent_reuse is not None:
            message = "ggn.cotangent_reuse applies only to GGNVP rows"
            raise MaterializationError(message)

        return

    if path == GGN_DENSE_PATH:
        if jvp_reuse is not None:
            message = "ggn.jvp_reuse is not used with dense_global"
            raise MaterializationError(message)

        if cotangent_reuse is not None:
            message = "ggn.cotangent_reuse is not used with dense_global"
            raise MaterializationError(message)

        return

    if jvp_reuse not in {None, "reuse_jvp", "recompute_jvp"}:
        message = f"ggn.jvp_reuse is unsupported: {jvp_reuse}"
        raise MaterializationError(message)

    if cotangent_reuse not in {
        None,
        "reuse_output_cotangent",
        "recompute_output_cotangent",
    }:
        message = f"ggn.cotangent_reuse is unsupported: {cotangent_reuse}"
        raise MaterializationError(message)


def _require_output_cotangent_block_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    block_size = _output_cotangent_block_size(settings)

    if block_size is None:
        return

    if operator.kind != "ggnvp":
        message = "chunk.output_cotangent_block_size applies only to GGNVP rows"
        raise MaterializationError(message)

    if path == GGN_DENSE_PATH:
        message = "chunk.output_cotangent_block_size requires a GGN VJP path"
        raise MaterializationError(message)


def _output_cotangent_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.output_cotangent_block_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _parameter_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.parameter_block_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _layer_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.layer_block_size"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        message = f"{key} must be a positive integer"
        raise MaterializationError(message)

    return value


def _require_parameter_block_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> None:
    block_size = _parameter_block_size(settings)
    layer_block_size = _layer_block_size(settings)

    if block_size is None and layer_block_size is None:
        return

    if block_size is not None and layer_block_size is not None:
        message = "parameter and layer chunk sizes cannot both be set"
        raise MaterializationError(message)

    if layer_block_size is not None:
        _layer_block_ranges(
            _parameter_surface_width(parameter_surface, "chunk.layer_block_size"),
            layer_block_size,
            parameter_surface,
        )

    if operator.kind == "ggnvp" and path == GGN_DENSE_PATH:
        return

    if operator.kind == "fisher_vp" and path == FISHER_DENSE_PATH:
        return

    if operator.kind == "sampled_fisher_vp" and path == SAMPLED_FISHER_DENSE_PATH:
        return

    if operator.kind == "empirical_fisher_vp" and path == EMPIRICAL_FISHER_DENSE_PATH:
        return

    message = "chunk.parameter_block_size requires a dense parameter-column matrix path"
    raise MaterializationError(message)


def _parameter_surface_width(
    parameter_surface: ParameterSurface | None,
    key: str,
) -> int:
    if parameter_surface is None:
        message = f"{key} requires declared layer_groups"
        raise MaterializationError(message)

    return sum(math.prod(shape) for shape in parameter_surface.shapes)


def _runtime_params(
    params: ParameterTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    _require_parameter_surface_runtime_settings(parameter_surface, settings)
    dtype = _parameter_dtype(settings)
    result = _runtime_named_tensor_dtype(params, dtype)
    result = _runtime_named_tensor_contiguity(result, settings)

    if settings.get("layout.params") == "flat_contiguous":
        _require_alias_safe_parameter_layout(result, settings)

        return _wrap_flat_parameter_tree(
            result,
            _flatten_vector(result).contiguous(),
        )

    if settings.get("layout.params") in {"per_layer_flat", "per_block_flat"}:
        _require_alias_safe_parameter_layout(result, settings)

    return _runtime_grouped_parameter_layout(
        result,
        settings,
        "layout.params",
        parameter_surface,
    )


def _runtime_grouped_parameter_layout(
    tree: ParameterTree,
    settings: Mapping[str, Any],
    key: str,
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    groups = _parameter_layout_groups(settings, key, parameter_surface)

    if groups is None:
        return tree

    return _wrap_grouped_parameter_tree(tree, groups, key)


def _parameter_layout_groups(
    settings: Mapping[str, Any],
    key: str,
    parameter_surface: ParameterSurface | None,
) -> tuple[tuple[str, ...], ...] | None:
    layout = settings.get(key)

    if layout in {None, "parameter_tree", "flat_contiguous"}:
        return None

    if layout == "per_layer_flat":
        return _declared_parameter_groups(parameter_surface, "layer_groups", key)

    if layout == "per_block_flat":
        return _declared_parameter_groups(parameter_surface, "block_groups", key)

    return None


def _declared_parameter_groups(
    parameter_surface: ParameterSurface | None,
    group_field: str,
    key: str,
) -> tuple[tuple[str, ...], ...]:
    if parameter_surface is None:
        message = f"{key} requires declared {group_field}"
        raise MaterializationError(message)

    groups = (
        parameter_surface.layer_groups
        if group_field == "layer_groups"
        else parameter_surface.block_groups
    )

    if not groups:
        message = f"{key} requires declared {group_field}"
        raise MaterializationError(message)

    return groups


def _wrap_grouped_parameter_tree(
    tree: ParameterTree,
    groups: tuple[tuple[str, ...], ...],
    key: str,
) -> ParameterTree:
    grouped = dict(tree)

    for group in groups:
        _require_group_names(tree, group, key)
        flat = torch.cat(tuple(tree[name].reshape(-1) for name in group)).contiguous()
        offset = 0

        for name in group:
            tensor = tree[name]
            stop = offset + tensor.numel()
            grouped[name] = flat[offset:stop].reshape_as(tensor)
            offset = stop

    return grouped


def _require_group_names(
    tree: ParameterTree,
    group: tuple[str, ...],
    key: str,
) -> None:
    missing = tuple(name for name in group if name not in tree)

    if missing:
        message = f"{key} declared groups contain missing parameters: {missing}"
        raise MaterializationError(message)


def _runtime_grouped_output_layout(
    tree: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    groups = _parameter_layout_groups(settings, "layout.output", parameter_surface)

    if groups is None:
        return tree

    result = _parameter_tree_from_tensor_tree(
        tree,
        "grouped output layout",
    )

    return _wrap_grouped_parameter_tree(result, groups, "layout.output")


def _parameter_tree_from_tensor_tree(
    tree: TensorTree,
    label: str,
) -> ParameterTree:
    if not isinstance(tree, dict):
        message = f"{label} requires a parameter-tree tensor mapping"
        raise MaterializationError(message)

    result = dict[str, torch.Tensor]()

    for key, value in tree.items():
        if not isinstance(key, str):
            message = f"{label} requires string keys"
            raise MaterializationError(message)

        if not isinstance(value, torch.Tensor):
            message = f"{label} requires tensor leaves"
            raise MaterializationError(message)

        result[key] = value

    return result


def _runtime_buffers(
    buffers: BufferTree,
    settings: Mapping[str, Any],
) -> BufferTree:
    dtype = _parameter_dtype(settings)

    if dtype is None:
        return _runtime_named_tensor_contiguity(buffers, settings)

    return _runtime_named_tensor_contiguity(
        {key: tensor.to(dtype=dtype) for key, tensor in buffers.items()},
        settings,
    )


def _runtime_batch(
    batch: Batch,
    settings: Mapping[str, Any],
    *,
    move_input_residency: bool = True,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> Batch:
    dtype = _batch_dtype(settings)
    metric_factor_dtype = _dtype_setting(settings, "dtype.metric_factor")
    metric_factor_residency = settings.get("memory.factor_residency")

    if (
        dtype is None
        and metric_factor_dtype is None
        and metric_factor_residency is None
    ):
        return _runtime_batch_after_contiguity(
            batch,
            settings,
            move_input_residency=move_input_residency,
        )

    result = (
        batch
        if dtype is None
        else {key: _runtime_batch_value(value, dtype) for key, value in batch.items()}
    )

    if metric_factor_dtype is None and metric_factor_residency is None:
        return _runtime_batch_after_contiguity(
            result,
            settings,
            move_input_residency=move_input_residency,
        )

    if metric_factor_dtype is not None:
        result = {
            key: _runtime_metric_factor_value(key, value, metric_factor_dtype)
            for key, value in result.items()
        }

    if metric_factor_residency is not None:
        result = {
            key: _runtime_metric_factor_residency_value(
                key,
                value,
                metric_factor_residency,
                mmap_residency,
            )
            for key, value in result.items()
        }

    return _runtime_batch_after_contiguity(
        result,
        settings,
        move_input_residency=move_input_residency,
    )


def _runtime_declared_batch_transforms(
    batch: Batch,
    candidate: Candidate,
    batch_layout: Callable[[Candidate, Batch], Batch] | None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None,
) -> Batch:
    result = batch

    if _uses_declared_batch_layout(candidate.settings):
        if batch_layout is None:
            message = "input batch layout requires declared binding"
            raise MaterializationError(message)

        result = batch_layout(candidate, result)

    if "chunk.lm_head_weight_chunk_bytes" in candidate.settings:
        if lm_head_chunker is None:
            message = "chunk.lm_head_weight_chunk_bytes requires LM-head weight binding"
            raise MaterializationError(message)

        result = lm_head_chunker(candidate, result)

    return result


def _uses_declared_batch_layout(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("input.batch_layout")
        in {"packed_with_inverse_permutation", "variable_length"}
        or settings.get("input.length_grouping") == "exact_length_bucket"
        or settings.get("schedule.per_token") == "packed"
    )


def _runtime_batch_after_contiguity(
    batch: Batch,
    settings: Mapping[str, Any],
    *,
    move_input_residency: bool,
) -> Batch:
    result = _runtime_batch_contiguity(batch, settings)

    if move_input_residency:
        result = _runtime_batch_input_residency(result, settings)

    return _runtime_batch_teacher_outputs(result, settings)


def _runtime_batch_input_residency(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    residency = settings.get("input.residency")

    if residency is None:
        return batch

    result = dict(batch)

    for key, value in batch.items():
        if key != "teacher_outputs":
            result[key] = _input_residency_value(value, residency)

    return result


def _input_residency_value(value: Any, residency: Any) -> Any:
    return _runtime_nested_tensor_value(
        value,
        lambda tensor: _input_residency_tensor(tensor, residency),
    )


def _input_residency_tensor(tensor: torch.Tensor, residency: Any) -> torch.Tensor:
    return _residency_tensor(tensor, residency, "input.residency")


def _tree_residency(tree: TensorTree, residency: Any, key: str) -> TensorTree:
    return tree_map(lambda tensor: _residency_tensor(tensor, residency, key), tree)


def _residency_tensor(
    tensor: torch.Tensor,
    residency: Any,
    key: str,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    if residency == "cpu_staged":
        return tensor.to(device=torch.device("cpu"))

    if residency == "cpu_pinned":
        cpu_tensor = tensor.to(device=torch.device("cpu"))

        try:
            return cpu_tensor.pin_memory()
        except RuntimeError as error:
            raise MaterializationError(str(error)) from error

    if residency == "gpu":
        if not torch.cuda.is_available():
            message = f"{key}=gpu requires CUDA"
            raise MaterializationError(message)

        return tensor.to(device=torch.device("cuda"))

    if residency == "mmap_cpu":
        if mmap_residency is None:
            message = f"{key}=mmap_cpu requires memory-mapped tensor metadata"
            raise MaterializationError(message)

        return mmap_residency(tensor, key)

    message = f"{key} is unsupported: {residency}"
    raise MaterializationError(message)


def _runtime_residency_tensor(
    tensor: torch.Tensor,
    residency: Any,
    key: str,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> torch.Tensor:
    if mmap_residency is None or residency != "mmap_cpu":
        return _residency_tensor(tensor, residency, key)

    return _residency_tensor(tensor, residency, key, mmap_residency)


def _runtime_batch_teacher_outputs(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    value = settings.get("teacher_outputs")

    if value is None:
        return batch

    if "teacher_outputs" not in batch:
        message = "teacher_outputs batch field is required"
        raise MaterializationError(message)

    result = dict(batch)

    if value == "precomputed_cpu":
        result["teacher_outputs"] = _teacher_outputs_to_device(
            batch["teacher_outputs"],
            torch.device("cpu"),
        )

        return result

    if value == "precomputed_cpu_pinned":
        result["teacher_outputs"] = _teacher_outputs_pin_cpu(batch["teacher_outputs"])

        return result

    if value == "precomputed_gpu":
        if not torch.cuda.is_available():
            message = "precomputed_gpu teacher outputs require CUDA"
            raise MaterializationError(message)

        result["teacher_outputs"] = _teacher_outputs_to_device(
            batch["teacher_outputs"],
            torch.device("cuda"),
        )

        return result

    if value == "recomputed_with_equality_check":
        _require_teacher_output_tree(batch["teacher_outputs"])

        return result

    message = f"teacher_outputs is unsupported: {value}"
    raise MaterializationError(message)


def _execution_with_recomputed_teacher_outputs(
    execution: StandardExecution,
) -> StandardExecution:
    if execution.candidate.settings.get("teacher_outputs") != (
        "recomputed_with_equality_check"
    ):
        return execution

    if execution.teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)

    fixed_outputs = execution.batch.get("teacher_outputs")
    _require_teacher_output_tree(fixed_outputs)
    settings = execution.candidate.settings
    recomputed_outputs = execution.teacher_objective(
        _model_compute_tree(execution.params, settings),
        _model_compute_tree(execution.buffers, settings),
        _model_compute_batch(execution.batch, settings),
        execution.context,
    )
    _require_teacher_outputs_match(fixed_outputs, recomputed_outputs)
    batch = dict(execution.batch)
    batch["teacher_outputs"] = recomputed_outputs

    return dataclasses.replace(execution, batch=batch)


def _teacher_outputs_to_device(value: Any, device: torch.device) -> TensorTree:
    return _runtime_nested_tensor_value(
        value,
        lambda tensor: tensor.to(device=device),
        error_message="teacher_outputs batch field must be a tensor tree",
    )


def _require_teacher_output_tree(value: Any) -> None:
    _runtime_nested_tensor_value(
        value,
        lambda tensor: tensor,
        error_message="teacher_outputs batch field must be a tensor tree",
    )


def _require_teacher_outputs_match(fixed: Any, recomputed: Any) -> None:
    if _teacher_outputs_equal(fixed, recomputed):
        return

    message = "recomputed teacher outputs do not match fixed teacher_outputs"
    raise MaterializationError(message)


def _teacher_outputs_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False

        return all(_teacher_outputs_equal(left[key], right[key]) for key in left)

    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False

        return all(starmap(_teacher_outputs_equal, zip(left, right, strict=True)))

    return False


def _teacher_outputs_pin_cpu(value: Any) -> TensorTree:
    cpu_value = _teacher_outputs_to_device(value, torch.device("cpu"))

    return tree_map(_pin_cpu_tensor, cpu_value)


def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        message = "teacher output pinning requires CPU tensors"
        raise MaterializationError(message)

    try:
        return tensor.pin_memory()
    except RuntimeError as error:
        raise MaterializationError(str(error)) from error


def _runtime_vector(
    vector: TensorTree,
    settings: Mapping[str, Any],
    template: TensorTree | None = None,
    parameter_surface: ParameterSurface | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> TensorTree:
    dtype = _dtype_setting(settings, "dtype.vector")

    if dtype is None:
        result = vector
    else:
        result = tree_map(lambda tensor: tensor.to(dtype=dtype), vector)

    result = _runtime_vector_residency(result, settings, mmap_residency)
    result = _runtime_vector_layout(
        result,
        settings,
        vector if template is None else template,
        parameter_surface,
    )

    return _runtime_tree_contiguity(result, settings)


def _runtime_vector_layout(
    vector: TensorTree,
    settings: Mapping[str, Any],
    template: TensorTree,
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    layout = settings.get("layout.vector")

    if layout is None or layout == "parameter_tree":
        return vector

    if isinstance(vector, torch.Tensor):
        flat_vector = vector.reshape(-1).contiguous()
    else:
        flat_vector = _flatten_vector_like(template, vector).contiguous()

    if layout == "flat_contiguous":
        return _wrap_flat_vector(template, flat_vector)

    wrapped = _wrap_flat_vector(template, flat_vector)
    parameter_tree = _parameter_tree_from_tensor_tree(
        wrapped,
        f"layout.vector={layout}",
    )

    return _runtime_grouped_parameter_layout(
        parameter_tree,
        settings,
        "layout.vector",
        parameter_surface,
    )


def _runtime_named_tensor_dtype(
    tree: dict[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    if dtype is None:
        return tree

    return _runtime_named_tensor_map_preserve_alias(
        tree,
        lambda tensor: tensor.to(dtype=dtype),
    )


def _runtime_named_tensor_map_preserve_alias(
    tree: dict[str, torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped = {}
    result = {}

    for key, tensor in tree.items():
        alias_key = id(tensor)

        if alias_key not in mapped:
            mapped[alias_key] = function(tensor)

        result[key] = mapped[alias_key]

    return result


def _require_alias_safe_parameter_layout(
    params: ParameterTree,
    settings: Mapping[str, Any],
) -> None:
    if not _preserves_parameter_aliases(settings):
        return

    if not _has_parameter_aliases(params):
        return

    message = "non-tree parameter layout cannot preserve tied parameter aliases"
    raise MaterializationError(message)


def _require_parameter_surface_runtime_settings(
    parameter_surface: ParameterSurface | None,
    settings: Mapping[str, Any],
) -> None:
    if parameter_surface is None:
        return

    if (
        _preserves_parameter_aliases(settings)
        and parameter_surface.tied_weights_policy != "preserve"
    ):
        message = "tied-weight preservation requires preserved parameter surface"
        raise MaterializationError(message)


def _preserves_parameter_aliases(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("call.tied_weights") == "preserve_alias_groups"
        or settings.get("layout.aliasing") == "preserve_tied_weight_aliases"
    )


def _has_parameter_aliases(params: ParameterTree) -> bool:
    ids = tuple(id(tensor) for tensor in params.values())

    return len(ids) != len(set(ids))


def _runtime_vector_residency(
    vector: TensorTree,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> TensorTree:
    residency = settings.get("memory.vector_residency")

    if residency is None:
        return vector

    return tree_map(
        lambda tensor: _runtime_residency_tensor(
            tensor,
            residency,
            "memory.vector_residency",
            mmap_residency,
        ),
        vector,
    )


def _runtime_output(
    output: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None = None,
) -> TensorTree:
    dtype = _dtype_setting(settings, "dtype.output")

    if dtype is not None:
        output = tree_map(lambda tensor: tensor.to(dtype=dtype), output)

    if _layout_output(settings) == "flat_contiguous":
        return _flatten_vector(output).contiguous()

    return _runtime_grouped_output_layout(output, settings, parameter_surface)


def _standard_output_buffer(execution: StandardExecution) -> TensorTree | None:
    if execution.candidate.settings.get("memory.output_buffers") != "preallocated":
        return None

    template = _standard_output_template(execution)
    runtime_template = _runtime_output(
        template,
        execution.candidate.settings,
        execution.parameter_surface,
    )

    return tree_map(torch.empty_like, runtime_template)


def _composition_output_buffer(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("memory.output_buffers") != "preallocated":
        return None

    template = _runtime_vector(vector, settings)
    runtime_template = _runtime_output(template, settings)

    return tree_map(torch.empty_like, runtime_template)


def _standard_output_template(execution: StandardExecution) -> TensorTree:
    kind = execution.operator.kind

    if kind == "jvp":
        return _jvp_output_template(execution)

    if kind in {
        "gradient",
        "vjp",
        "hvp",
        "ggnvp",
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return execution.params

    if kind in {"metric", "inverse_metric", "composition"}:
        return execution.vector

    message = f"memory.output_buffers=preallocated lacks output template for {kind}"
    raise MaterializationError(message)


def _jvp_output_template(execution: StandardExecution) -> TensorTree:
    function = _function_objective(
        execution.operator,
        execution.function_objectives,
    )

    def callback() -> TensorTree:
        return _call_function_objective(execution, function, execution.params)

    return _run_with_backend_settings(
        execution.candidate.settings,
        lambda: _run_with_call_grad_mode(execution.candidate.settings, callback),
    )


def _runtime_output_to_buffer(
    output: TensorTree,
    buffer: TensorTree | None,
) -> TensorTree:
    if buffer is None:
        return output

    try:
        return tree_map2(_copy_output_tensor, buffer, output)
    except (RuntimeError, TypeError) as error:
        message = "memory.output_buffers=preallocated output tree mismatch"
        raise MaterializationError(message) from error


def _copy_output_tensor(buffer: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if buffer.dtype != output.dtype:
        message = "preallocated output buffer dtype mismatch"
        raise RuntimeError(message)

    if buffer.device != output.device:
        message = "preallocated output buffer device mismatch"
        raise RuntimeError(message)

    buffer.copy_(output)

    return buffer


def _accumulation_tensor(
    tensor: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    dtype = _dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tensor

    return tensor.to(dtype=dtype)


def _accumulation_tree(tree: TensorTree, settings: Mapping[str, Any]) -> TensorTree:
    dtype = _dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tree

    return tree_map(lambda tensor: tensor.to(dtype=dtype), tree)


def _runtime_batch_value(value: Any, dtype: torch.dtype) -> Any:
    def convert(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(dtype=dtype)

        return tensor

    return _runtime_nested_tensor_value(value, convert)


def _runtime_nested_tensor_value(
    value: Any,
    map_tensor: Callable[[torch.Tensor], Any],
    *,
    error_message: str | None = None,
) -> Any:
    if isinstance(value, torch.Tensor):
        return map_tensor(value)

    if isinstance(value, dict):
        return {
            key: _runtime_nested_tensor_value(
                child,
                map_tensor,
                error_message=error_message,
            )
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _runtime_nested_tensor_value(
                child,
                map_tensor,
                error_message=error_message,
            )
            for child in value
        )

    if error_message is not None:
        raise MaterializationError(error_message)

    return value


def _runtime_metric_factor_value(
    key: str,
    value: Any,
    dtype: torch.dtype,
) -> Any:
    if key not in METRIC_FACTOR_BATCH_KEYS:
        return value

    return _runtime_batch_value(value, dtype)


def _runtime_metric_factor_residency_value(
    key: str,
    value: Any,
    residency: Any,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> Any:
    if key not in METRIC_FACTOR_BATCH_KEYS:
        return value

    return _runtime_nested_tensor_value(
        value,
        lambda tensor: _runtime_residency_tensor(
            tensor,
            residency,
            "memory.factor_residency",
            mmap_residency,
        ),
    )


def _runtime_tree_contiguity(
    tree: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    if not _layout_contiguity_enabled(settings):
        return tree

    return tree_map(lambda tensor: tensor.contiguous(), tree)


def _runtime_named_tensor_contiguity(
    tree: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if not _layout_contiguity_enabled(settings):
        return tree

    return _runtime_named_tensor_map_preserve_alias(
        tree,
        lambda tensor: tensor.contiguous(),
    )


def _runtime_batch_contiguity(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    if not _layout_contiguity_enabled(settings):
        return batch

    return {
        key: _runtime_nested_tensor_value(value, lambda tensor: tensor.contiguous())
        for key, value in batch.items()
    }


def _layout_contiguity_enabled(settings: Mapping[str, Any]) -> bool:
    value = settings.get("layout.contiguity")

    if value is None or value == "preserve_existing_strides":
        return False

    if value == "contiguous":
        return True

    message = f"layout.contiguity is unsupported: {value}"
    raise MaterializationError(message)


def _parameter_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    autodiff_dtype = _dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return _dtype_setting(settings, "dtype.parameter_storage")


def _batch_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    autodiff_dtype = _dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return _dtype_setting(settings, "dtype.intermediate")


def _dtype_setting(settings: Mapping[str, Any], key: str) -> torch.dtype | None:
    dtype_name = settings.get(key)

    if dtype_name is None:
        return None

    if not isinstance(dtype_name, str):
        message = f"{key} must be a string"
        raise MaterializationError(message)

    if dtype_name == "bf16":
        return torch.bfloat16

    if dtype_name == "fp16":
        return torch.float16

    if dtype_name == "fp32":
        return torch.float32

    if dtype_name == "fp8_when_supported":
        return _fp8_dtype()

    message = f"{key} is unsupported by standard runtime: {dtype_name}"
    raise MaterializationError(message)


def _fp8_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn

    message = "fp8_when_supported requires PyTorch FP8 dtype support"
    raise MaterializationError(message)


def _run_with_backend_settings(
    settings: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    if not BACKEND_SETTINGS_ENABLED[0]:
        return callback()

    matmul_precision = _matmul_precision_setting(settings)
    autocast_setting = _autocast_setting(settings)
    allow_bf16_reduction = _bool_string_setting(
        settings,
        "numeric.bf16_reduced_precision_reduction",
    )
    allow_fp16_reduction = _bool_string_setting(
        settings,
        "numeric.fp16_reduced_precision_reduction",
    )
    deterministic_algorithms = _bool_string_setting(
        settings,
        "numeric.deterministic_algorithms",
    )
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_allow_bf16_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    previous_allow_fp16_reduction = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )
    previous_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)

    if allow_bf16_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            allow_bf16_reduction
        )

    if allow_fp16_reduction is not None:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            allow_fp16_reduction
        )

    if deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(deterministic_algorithms)

    try:
        if autocast_setting is None:
            return callback()

        device_type, dtype = autocast_setting

        with torch.autocast(device_type=device_type, dtype=dtype):
            return callback()
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_allow_bf16_reduction
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            previous_allow_fp16_reduction
        )
        torch.use_deterministic_algorithms(previous_deterministic_algorithms)


def _run_with_call_grad_mode(
    settings: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    value = settings.get("call.grad_mode")

    if value is None:
        return callback()

    if value == "grad_enabled":
        with torch.enable_grad():
            return callback()

    message = f"call.grad_mode is unsupported: {value}"
    raise MaterializationError(message)


def _run_with_buffer_mutation_check(
    execution: StandardExecution,
    callback: CandidateOperation,
) -> TensorTree:
    mode = execution.candidate.settings.get("call.buffer_mutation")

    if mode is None:
        return callback()

    if mode == "declared_and_restored":
        return _run_with_declared_state_restore(execution, callback)

    if mode != "forbidden":
        message = f"call.buffer_mutation is unsupported: {mode}"
        raise MaterializationError(message)

    before = _buffer_snapshot(execution.buffers)
    result = callback()
    _require_buffers_unchanged(before, execution.buffers)

    return result


def _run_with_declared_state_restore(
    execution: StandardExecution,
    callback: CandidateOperation,
) -> TensorTree:
    settings = execution.candidate.settings
    parameter_snapshot = _declared_tensor_snapshot(
        execution.params,
        settings["mutated_parameter_keys"],
        "parameter",
    )
    buffer_snapshot = _declared_tensor_snapshot(
        execution.buffers,
        settings["mutated_buffer_keys"],
        "buffer",
    )

    try:
        return callback()
    finally:
        _restore_declared_tensors(execution.params, parameter_snapshot)
        _restore_declared_tensors(execution.buffers, buffer_snapshot)


def _declared_tensor_snapshot(
    values: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    label: str,
) -> dict[str, torch.Tensor]:
    snapshot = {}

    for key in keys:
        if key not in values:
            message = f"declared mutated {label} is missing: {key}"
            raise MaterializationError(message)

        snapshot[key] = values[key].detach().clone()

    return snapshot


def _restore_declared_tensors(
    values: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for key, tensor in snapshot.items():
            values[key].copy_(tensor)


def _buffer_snapshot(buffers: BufferTree) -> BufferTree:
    return {key: tensor.detach().clone() for key, tensor in buffers.items()}


def _require_buffers_unchanged(before: BufferTree, after: BufferTree) -> None:
    if set(before) != set(after):
        message = "call.buffer_mutation=forbidden detected changed buffer keys"
        raise MaterializationError(message)

    for key, before_tensor in before.items():
        after_tensor = after[key]

        if before_tensor.shape != after_tensor.shape:
            message = f"call.buffer_mutation=forbidden changed buffer shape: {key}"
            raise MaterializationError(message)

        if before_tensor.dtype != after_tensor.dtype:
            message = f"call.buffer_mutation=forbidden changed buffer dtype: {key}"
            raise MaterializationError(message)

        if before_tensor.device != after_tensor.device:
            message = f"call.buffer_mutation=forbidden changed buffer device: {key}"
            raise MaterializationError(message)

        if not torch.equal(before_tensor, after_tensor):
            message = f"call.buffer_mutation=forbidden changed buffer value: {key}"
            raise MaterializationError(message)


def _matmul_precision_setting(settings: Mapping[str, Any]) -> str | None:
    key = "numeric.float32_matmul_precision"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        message = f"{key} must be a string"
        raise MaterializationError(message)

    if value not in {"highest", "high", "medium"}:
        message = f"{key} is unsupported by standard runtime: {value}"
        raise MaterializationError(message)

    return value


def _autocast_setting(settings: Mapping[str, Any]) -> tuple[str, torch.dtype] | None:
    value = settings.get("autocast")

    if value is None or value == "off":
        return None

    if value == "cuda_fp16":
        return _cuda_autocast(torch.float16)

    if value == "cuda_bf16":
        return _cuda_autocast(torch.bfloat16)

    message = f"autocast is unsupported by standard runtime: {value}"
    raise MaterializationError(message)


def _cuda_autocast(dtype: torch.dtype) -> tuple[str, torch.dtype]:
    if not torch.cuda.is_available():
        message = "CUDA autocast requires CUDA"
        raise MaterializationError(message)

    return "cuda", dtype


def _bool_string_setting(settings: Mapping[str, Any], key: str) -> bool | None:
    value = settings.get(key)

    if value is None:
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be true or false"
    raise MaterializationError(message)


def _anchor_candidate(operator: OperatorSpec, candidate: Candidate) -> Candidate:
    path = _anchor_path(operator)
    settings = _anchor_settings(operator, candidate, path)

    return dataclasses.replace(
        candidate,
        settings=settings,
    )


def _anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    settings = dict(candidate.settings)

    for key in (
        *RUNTIME_DTYPE_SETTINGS,
        *BACKEND_SETTINGS,
        *SPEC_PATH_KEYS.values(),
        *SPEC_ADDITIONAL_RUNTIME_SETTINGS,
        *FUNCTIONAL_CALL_FIELDS,
        *TORCH_FUNC_FIELDS,
        *LOSS_SCALING_SETTINGS,
        "vectorization.vmap_chunk_size",
        "vectorization.in_dims",
    ):
        settings.pop(key, None)

    spec_value = _spec_path_value_for_runtime_path(operator.kind, path)

    if spec_value is None:
        if operator.kind != "ggnvp" or path != GGN_DENSE_PATH:
            message = f"anchor path has no SPEC mapping: {path}"
            raise MaterializationError(message)
    else:
        settings[SPEC_PATH_KEYS[operator.kind]] = spec_value

    settings.update(_fisher_anchor_settings(operator, path))
    settings.update(_sampled_fisher_anchor_settings(operator, candidate, path))
    settings.update(_per_example_gradient_anchor_settings(operator, path))
    settings.update(_ggn_anchor_settings(operator, path))

    settings.update(_anchor_admission_settings(path))

    return settings


def _fisher_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "fisher_vp":
        return {}

    if path != FISHER_SCORE_GRADIENT_LOOP_PATH:
        return {}

    return {
        "fisher.expectation_path": "explicit_full_expectation_score_rows",
        "fisher.score_grad_path": "torch_autograd_grad_loop",
    }


def _sampled_fisher_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "sampled_fisher_vp":
        return {}

    if path != SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH:
        return {}

    sample_source_key = "sampled_fisher.sample_source"
    exact_check_key = "sampled_fisher.exact_fisher_check"
    settings = {
        "sampled_fisher.score_grad_path": "torch_autograd_grad_loop",
    }

    for key in (sample_source_key, exact_check_key):
        if key not in candidate.settings:
            message = f"{key} is required for sampled Fisher anchors"
            raise MaterializationError(message)

        settings[key] = candidate.settings[key]

    return settings


def _per_example_gradient_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "per_example_gradient":
        return {}

    if path != PER_EXAMPLE_GRADIENT_LOOP_PATH:
        return {}

    return {"per_example_gradient.accumulation": "stacked_leading_axis"}


def _ggn_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "ggnvp":
        return {}

    if path == GGN_DENSE_PATH:
        return {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
        }

    if path == GGN_JVP_HESSIAN_VJP_PATH:
        return {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
            "ggn.vjp_path": "torch_func_vjp",
        }

    return {}


def _spec_path_value_for_runtime_path(operator_kind: str, path: str) -> str | None:
    path_map = SPEC_PATH_TO_RUNTIME.get(operator_kind, {})

    for spec_value, runtime_path in path_map.items():
        if runtime_path == path:
            return spec_value

    return None


def _anchor_admission_settings(path: str) -> dict[str, Any]:
    if path == JVP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    if path == VJP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=False)

    if path == HVP_JVP_GRAD_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    if path == GGN_JVP_HESSIAN_VJP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    return {}


def _torch_func_anchor_settings(*, requires_forward_ad: bool) -> dict[str, Any]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def _anchor_path(operator: OperatorSpec) -> str:
    if operator.kind == "fisher_vp":
        return _fisher_anchor_path(operator)

    path = STANDARD_ANCHOR_PATHS.get(operator.kind)

    if path is not None:
        return path

    message = f"standard anchor does not support operator kind: {operator.kind}"
    raise MaterializationError(message)


def _fisher_anchor_path(operator: OperatorSpec) -> str:
    distribution = _operator_semantic(operator, "distribution")

    if distribution == "explicit_score_gradients":
        return FISHER_SCORE_GRADIENT_LOOP_PATH

    message = "standard Fisher anchor does not support declared semantics"
    raise MaterializationError(message)


def _require_candidate_family(operator: OperatorSpec, candidate: Candidate) -> None:
    if candidate.family != operator.family:
        message = (
            f"candidate family does not match operator family: "
            f"{candidate.family} != {operator.family}"
        )
        raise MaterializationError(message)


def _require_path(
    operator_kind: str,
    path: str,
    allowed_paths: tuple[str, ...],
) -> None:
    if path in allowed_paths:
        return

    message = f"operator path {path} does not support {operator_kind}"
    raise MaterializationError(message)


def _scalar_objective(
    operator: OperatorSpec,
    scalar_objectives: Mapping[str, ScalarObjective],
) -> ScalarObjective:
    objective = scalar_objectives.get(operator.objective_id)

    if objective is None:
        message = f"scalar objective is missing: {operator.objective_id}"
        raise MaterializationError(message)

    return objective


def _function_objective(
    operator: OperatorSpec,
    function_objectives: Mapping[str, FunctionObjective],
) -> FunctionObjective:
    objective = function_objectives.get(operator.objective_id)

    if objective is None:
        message = f"function objective is missing: {operator.objective_id}"
        raise MaterializationError(message)

    return objective


def _call_function_objective(
    execution: StandardExecution,
    function: FunctionObjective,
    params: ParameterTree,
    batch: Batch | None = None,
) -> TensorTree:
    active_batch = execution.batch if batch is None else batch
    settings = execution.candidate.settings
    output = function(
        _model_compute_tree(params, settings),
        _model_compute_tree(execution.buffers, settings),
        _model_compute_batch(active_batch, settings),
        execution.context,
    )

    return _checked_function_output(
        execution.candidate.settings,
        output,
        "function objective output",
    )


def _checked_function_output(
    settings: Mapping[str, Any],
    output: object,
    name: str,
) -> TensorTree:
    if settings.get("call.return_type") != "raw_tensor_tree":
        if not _is_raw_tensor_tree(output):
            message = f"{name} must be a tensor tree"
            raise MaterializationError(message)

        return output

    if not _is_raw_tensor_tree(output):
        message = f"{name} must be a raw tensor tree"
        raise MaterializationError(message)

    return output


def _is_raw_tensor_tree(output: object) -> TypeGuard[TensorTree]:
    if isinstance(output, torch.Tensor):
        return True

    if isinstance(output, tuple):
        return all(_is_raw_tensor_tree(value) for value in output)

    if isinstance(output, dict):
        return all(
            isinstance(key, str) and _is_raw_tensor_tree(value)
            for key, value in output.items()
        )

    return False


def _require_metric_representation(
    operator: OperatorSpec,
    allowed: tuple[str, ...],
) -> None:
    kind = _metric_representation_kind(operator)

    if kind not in allowed:
        message = f"metric representation kind is not supported by path: {kind}"
        raise MaterializationError(message)


def _metric_representation_kind(operator: OperatorSpec) -> str:
    representation = _metric_representation(operator)
    kind = representation.get("kind")

    if not isinstance(kind, str):
        message = "metric representation kind is required"
        raise MaterializationError(message)

    return kind


def _metric_representation(operator: OperatorSpec) -> Mapping[str, Any]:
    representation = operator.semantics.get("representation")

    if not isinstance(representation, Mapping):
        message = "metric representation is required"
        raise MaterializationError(message)

    return representation


def _matrix_free_metric_product(operator: OperatorSpec) -> str:
    representation = _metric_representation(operator)
    product = representation.get("operator")

    if not isinstance(product, str) or not product:
        message = "matrix_free metric requires a named sibling product"
        raise MaterializationError(message)

    return product


def _grad_enabled_params(params: ParameterTree) -> ParameterTree:
    return {
        name: tensor.detach().requires_grad_(True) for name, tensor in params.items()
    }


def _parameter_grad_tree(params: ParameterTree) -> TensorTree:
    return tree_from_leaves(
        params,
        tuple(
            torch.zeros_like(param) if param.grad is None else param.grad.detach()
            for param in params.values()
        ),
    )


def _flatten_vector(vector: TensorTree) -> torch.Tensor:
    leaves = tree_leaves(vector)

    if not leaves:
        message = "dense standard operator requires at least one tensor leaf"
        raise MaterializationError(message)

    return torch.cat(tuple(leaf.reshape(-1) for leaf in leaves))


def _flatten_vector_like(template: TensorTree, vector: TensorTree) -> torch.Tensor:
    if isinstance(vector, torch.Tensor):
        return vector.reshape(-1)

    if isinstance(template, torch.Tensor):
        message = "flat tensor template requires a flat tensor vector"
        raise MaterializationError(message)

    leaves = _matching_vector_leaves(template, vector)

    return torch.cat(tuple(leaf.reshape(-1) for leaf in leaves))


def _matching_vector_leaves(
    params: TensorTree, vector: TensorTree
) -> tuple[torch.Tensor, ...]:
    checked = tree_map2(
        lambda param, tangent: tangent.reshape_as(param), params, vector
    )

    return tree_leaves(checked)


def _wrap_flat_parameter_tree(
    template: ParameterTree,
    result: torch.Tensor,
) -> ParameterTree:
    leaves = []
    offset = 0

    for leaf in template.values():
        width = leaf.numel()
        leaves.append(result[offset : offset + width].reshape_as(leaf))
        offset += width

    if offset != result.numel():
        message = "flat parameter layout length differs from parameter tree"
        raise MaterializationError(message)

    return dict(zip(template, leaves, strict=True))


def _flatten_vector_batch(
    template: TensorTree,
    vector: TensorTree,
    in_dims: Any,
) -> torch.Tensor:
    batch_size = _vector_tree_batch_size(vector, in_dims)
    pieces = []
    _collect_flat_vector_batch_pieces(
        template,
        vector,
        in_dims,
        batch_size,
        pieces,
    )

    return torch.cat(tuple(pieces), dim=1)


def _collect_flat_vector_batch_pieces(
    template: TensorTree,
    vector: TensorTree,
    in_dims: Any,
    batch_size: int,
    pieces: list[torch.Tensor],
) -> None:
    if isinstance(template, torch.Tensor) and isinstance(vector, torch.Tensor):
        pieces.append(_flat_vector_batch_piece(template, vector, in_dims, batch_size))

        return

    if _is_tensor_tree_dict(template) and _is_tensor_tree_dict(vector):
        if not isinstance(in_dims, Mapping) or set(in_dims) != set(template):
            message = "vectorization.in_dims must match the vector tree"
            raise MaterializationError(message)

        if set(vector) != set(template):
            message = "vector tree mapping keys differ from parameters"
            raise MaterializationError(message)

        for key in template:
            _collect_flat_vector_batch_pieces(
                template[key],
                vector[key],
                in_dims[key],
                batch_size,
                pieces,
            )

        return

    if _is_tensor_tree_tuple(template) and _is_tensor_tree_tuple(vector):
        if not isinstance(in_dims, tuple) or len(in_dims) != len(template):
            message = "vectorization.in_dims must match the vector tree"
            raise MaterializationError(message)

        if len(vector) != len(template):
            message = "vector tree sequence length differs from parameters"
            raise MaterializationError(message)

        for param_leaf, vector_leaf, in_dim in zip(
            template,
            vector,
            in_dims,
            strict=True,
        ):
            _collect_flat_vector_batch_pieces(
                param_leaf,
                vector_leaf,
                in_dim,
                batch_size,
                pieces,
            )

        return

    message = "vector tree structure differs from parameters"
    raise MaterializationError(message)


def _flat_vector_batch_piece(
    template: torch.Tensor,
    vector: torch.Tensor,
    in_dim: Any,
    batch_size: int,
) -> torch.Tensor:
    if in_dim is None:
        if vector.shape != template.shape:
            message = "unmapped vector leaf shape differs from parameter leaf"
            raise MaterializationError(message)

        return vector.reshape(1, -1).expand(batch_size, -1)

    if not isinstance(in_dim, int) or isinstance(in_dim, bool):
        message = "vectorization.in_dims values must be integers or None"
        raise MaterializationError(message)

    dim = _normalized_vector_dim(vector, in_dim)
    unbatched_shape = vector.shape[:dim] + vector.shape[dim + 1 :]

    if unbatched_shape != template.shape:
        message = "mapped vector leaf shape differs from parameter leaf"
        raise MaterializationError(message)

    if vector.shape[dim] != batch_size:
        message = "vectorized vector mapped dimensions differ"
        raise MaterializationError(message)

    return vector.movedim(dim, 0).reshape(batch_size, -1)


def _wrap_flat_vector(template: TensorTree, result: torch.Tensor) -> TensorTree:
    leaves = []
    offset = 0

    for leaf in tree_leaves(template):
        width = leaf.numel()
        leaves.append(result[offset : offset + width].reshape_as(leaf))
        offset += width

    if offset != result.numel():
        message = "dense standard operator output length differs from vector tree"
        raise MaterializationError(message)

    return tree_from_leaves(template, tuple(leaves))


def _wrap_flat_vector_batch(template: TensorTree, result: torch.Tensor) -> TensorTree:
    if result.ndim != MATRIX_DIMS:
        message = "batched flat vector result must be a matrix"
        raise MaterializationError(message)

    leaves = []
    offset = 0

    for leaf in tree_leaves(template):
        width = leaf.numel()
        leaves.append(
            result[:, offset : offset + width].reshape(result.shape[0], *leaf.shape)
        )
        offset += width

    if offset != result.shape[1]:
        message = "batched dense output width differs from parameter tree"
        raise MaterializationError(message)

    return tree_from_leaves(template, tuple(leaves))


def _batch_tensor(batch: Batch, key: str) -> torch.Tensor:
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        message = f"batch tensor is missing: {key}"
        raise MaterializationError(message)

    return value


def _batch_tensor_blocks(batch: Batch, key: str) -> tuple[torch.Tensor, ...]:
    value = batch.get(key)

    if not isinstance(value, tuple) or not value:
        message = f"batch tensor blocks are missing: {key}"
        raise MaterializationError(message)

    row_count = None

    for block in value:
        if not isinstance(block, torch.Tensor):
            message = f"batch tensor block must be a tensor: {key}"
            raise MaterializationError(message)

        if block.ndim != MATRIX_DIMS:
            message = f"batch tensor block must be two-dimensional: {key}"
            raise MaterializationError(message)

        if block.shape[0] == 0:
            message = f"batch tensor blocks must have at least one row: {key}"
            raise MaterializationError(message)

        _require_finite_tensor(block, key)

        if row_count is None:
            row_count = block.shape[0]
        elif block.shape[0] != row_count:
            message = f"batch tensor blocks must share row count: {key}"
            raise MaterializationError(message)

    return value


def _batch_tree(batch: Batch, key: str) -> TensorTree:
    value = batch.get(key)

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, tuple):
        tree_leaves(value)

        return value

    if isinstance(value, dict):
        tree_leaves(value)

        return value

    message = f"batch tensor tree is missing: {key}"
    raise MaterializationError(message)


def _normalization(batch: Batch, operator: OperatorSpec) -> float:
    value = batch.get("normalization")

    if not isinstance(value, int | float):
        message = "batch normalization is missing"
        raise MaterializationError(message)

    normalization = float(value)

    if normalization <= 0.0:
        message = "batch normalization must be positive"
        raise MaterializationError(message)

    if operator.aggregation == "sum" and not math.isclose(
        normalization,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        message = "sum aggregation requires normalization=1.0"
        raise MaterializationError(message)

    return normalization


def _empirical_fisher_normalization(
    batch: Batch,
    operator: OperatorSpec,
    per_example_gradients: torch.Tensor,
) -> float:
    return _empirical_fisher_normalization_from_count(
        batch,
        operator,
        per_example_gradients.shape[0],
    )


def _empirical_fisher_streaming_normalization(
    execution: StandardExecution,
) -> float:
    batch, batch_in_dims = _per_example_batch_in_dims(
        execution.batch,
        "empirical Fisher streaming",
    )
    example_count = _per_example_batch_size(
        batch,
        batch_in_dims,
        "empirical Fisher streaming",
    )

    return _empirical_fisher_normalization_from_count(
        execution.batch,
        execution.operator,
        example_count,
    )


def _empirical_fisher_normalization_from_count(
    batch: Batch,
    operator: OperatorSpec,
    example_count: int,
) -> float:
    _require_empirical_fisher_semantics(operator)
    denominator = _operator_semantic(operator, "denominator")

    if denominator == "num_examples":
        normalization = float(example_count)
    elif denominator == "one":
        normalization = 1.0
    elif denominator == "batch_normalization":
        normalization = _normalization(batch, operator)
    else:
        message = f"empirical Fisher denominator is unsupported: {denominator}"
        raise MaterializationError(message)

    if operator.aggregation == "sum" and not math.isclose(
        normalization,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        message = "sum aggregation requires empirical Fisher denominator one"
        raise MaterializationError(message)

    return normalization


def _require_empirical_fisher_semantics(operator: OperatorSpec) -> None:
    example_loss_reduction = _operator_semantic(operator, "example_loss_reduction")

    if example_loss_reduction != "per_example":
        message = (
            "empirical Fisher example_loss_reduction is unsupported: "
            f"{example_loss_reduction}"
        )
        raise MaterializationError(message)


def _fisher_normalization(execution: StandardExecution) -> float:
    denominator = _operator_semantic(execution.operator, "denominator")

    if denominator == "num_examples":
        value = execution.batch.get("num_examples")

        if not isinstance(value, int | float):
            message = "num_examples denominator requires batch num_examples"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_examples denominator must be positive"
            raise MaterializationError(message)

        return float(value)

    if denominator == "one":
        return 1.0

    if denominator == "batch_normalization":
        return _normalization(execution.batch, execution.operator)

    message = f"Fisher denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _sampled_fisher_normalization(execution: StandardExecution) -> float:
    denominator = _operator_semantic(execution.operator, "denominator")
    sample_count = _operator_semantic_positive_int(execution.operator, "sample_count")

    if denominator == "num_examples":
        value = execution.batch.get("num_examples")

        if not isinstance(value, int | float):
            message = "num_examples denominator requires batch num_examples"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_examples denominator must be positive"
            raise MaterializationError(message)

        return float(value) * float(sample_count)

    if denominator == "num_tokens":
        value = execution.batch.get("num_tokens")

        if not isinstance(value, int | float):
            message = "num_tokens denominator requires batch num_tokens"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_tokens denominator must be positive"
            raise MaterializationError(message)

        return float(value) * float(sample_count)

    if denominator == "one":
        return 1.0

    if denominator == "batch_normalization":
        return _normalization(execution.batch, execution.operator)

    message = f"sampled Fisher denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _check_sampled_fisher_exact_bound(
    execution: StandardExecution,
    result: torch.Tensor,
) -> None:
    exact_check = execution.candidate.settings.get("sampled_fisher.exact_fisher_check")

    if exact_check == "disabled":
        return

    if exact_check != "enabled_with_sampling_bound":
        message = f"sampled_fisher.exact_fisher_check is unsupported: {exact_check}"
        raise MaterializationError(message)

    bound = _sampled_fisher_sampling_bound(execution.operator)
    exact = _batch_tensor(execution.batch, "exact_fisher_vp").reshape(-1)
    _require_finite_tensor(exact, "exact FisherVP reference")

    if exact.numel() != result.numel():
        message = "exact_fisher_vp must match sampled Fisher result width"
        raise MaterializationError(message)

    difference = (result.reshape(-1) - exact).norm()
    exact_norm = exact.norm()
    floor = torch.tensor(
        bound["norm_floor"],
        dtype=exact_norm.dtype,
        device=exact_norm.device,
    )
    relative = difference / torch.maximum(exact_norm, floor)
    abs_error = float(difference.item())
    rel_error = float(relative.item())

    if abs_error <= bound["max_abs_diff"] or rel_error <= bound["max_rel_diff"]:
        return

    message = (
        "sampled Fisher exact-Fisher comparison exceeded bound: "
        f"abs={abs_error}, rel={rel_error}"
    )
    raise MaterializationError(message)


def _sampled_fisher_sampling_bound(operator: OperatorSpec) -> dict[str, float]:
    raw = operator.semantics.get("sampling_bound")

    if not isinstance(raw, Mapping):
        message = "sampled Fisher sampling_bound must be a mapping"
        raise MaterializationError(message)

    if raw.get("kind") != "abs_or_rel":
        message = "sampled Fisher sampling_bound.kind must be abs_or_rel"
        raise MaterializationError(message)

    return {
        "max_abs_diff": _sampling_bound_float(raw, "max_abs_diff"),
        "max_rel_diff": _sampling_bound_float(raw, "max_rel_diff"),
        "norm_floor": _sampling_bound_float(raw, "norm_floor"),
    }


def _sampling_bound_float(bound: Mapping[str, Any], key: str) -> float:
    value = bound.get(key)

    if not isinstance(value, int | float):
        message = f"sampled Fisher sampling_bound.{key} must be numeric"
        raise MaterializationError(message)

    result = float(value)

    if result < 0.0:
        message = f"sampled Fisher sampling_bound.{key} must be nonnegative"
        raise MaterializationError(message)

    return result


def _require_sampled_fisher_semantics(execution: StandardExecution) -> None:
    operator = execution.operator
    _operator_semantic_positive_int(operator, "sample_count")
    operator_sample_source = _operator_semantic(operator, "sample_source")
    row_sample_source = execution.candidate.settings.get("sampled_fisher.sample_source")

    if row_sample_source not in {"fixed_sample_table", "fixed_seed_and_count"}:
        message = "sampled_fisher.sample_source is required"
        raise MaterializationError(message)

    if row_sample_source != operator_sample_source:
        message = "sampled_fisher.sample_source differs from operator"
        raise MaterializationError(message)

    exact_check = execution.candidate.settings.get("sampled_fisher.exact_fisher_check")

    if exact_check is None:
        message = "sampled_fisher.exact_fisher_check is required"
        raise MaterializationError(message)

    if exact_check not in {"disabled", "enabled_with_sampling_bound"}:
        message = f"sampled_fisher.exact_fisher_check is unsupported: {exact_check}"
        raise MaterializationError(message)

    score_reduction = _operator_semantic(operator, "score_reduction")

    if score_reduction != "none":
        message = f"sampled Fisher score_reduction is unsupported: {score_reduction}"
        raise MaterializationError(message)


def _require_fisher_semantics(
    operator: OperatorSpec,
    required: Mapping[str, str],
) -> None:
    for key, expected in required.items():
        actual = _operator_semantic(operator, key)

        if actual != expected:
            message = f"Fisher semantic field mismatch: {key}"
            raise MaterializationError(message)


def _require_valid_fisher_semantics(operator: OperatorSpec) -> None:
    distribution = _operator_semantic(operator, "distribution")

    if distribution == "explicit_score_gradients":
        _require_explicit_score_fisher_semantics(operator)

        return

    message = f"Fisher distribution is unsupported: {distribution}"
    raise MaterializationError(message)


def _require_explicit_score_fisher_semantics(operator: OperatorSpec) -> None:
    _require_fisher_semantics(
        operator,
        {
            "distribution": "explicit_score_gradients",
            "label_policy": "explicit_scores",
            "sample_space": "terms",
            "score_reduction": "none",
        },
    )


FISHER_FAMILY_RUNTIME_ROWS = {
    "fisher_vp": {
        "paths": FISHER_VECTOR_VMAP_PATHS,
        "streaming_paths": FISHER_SCORE_GRADIENT_PRODUCT_PATHS,
        "blockwise_path": FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        "dense_path": FISHER_DENSE_PATH,
        "block_batch_key": "score_gradient_blocks",
        "matrix_batch_key": "score_gradients",
        "streaming_matrix": _fisher_score_gradients,
        "streaming_normalization": _fisher_normalization,
        "blockwise_normalization": _fisher_normalization,
        "matrix_normalization": _fisher_score_matrix_normalization,
        "streaming_label": "Fisher",
        "vector_label": "Fisher vector",
        "result_label": "Fisher result",
        "blockwise_label": "score_gradients",
        "require_streaming": _require_explicit_score_fisher_execution,
        "require_matrix": _require_valid_fisher_execution,
        "check_result": _skip_score_matrix_result_check,
        "vmap_error_message": (
            "Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _skip_score_fisher_requirement,
    },
    "sampled_fisher_vp": {
        "paths": SAMPLED_FISHER_VECTOR_VMAP_PATHS,
        "streaming_paths": SAMPLED_FISHER_SCORE_GRADIENT_PRODUCT_PATHS,
        "blockwise_path": SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        "dense_path": SAMPLED_FISHER_DENSE_PATH,
        "block_batch_key": "sampled_score_gradient_blocks",
        "matrix_batch_key": "sampled_score_gradients",
        "streaming_matrix": _sampled_fisher_score_gradients,
        "streaming_normalization": _sampled_fisher_normalization,
        "blockwise_normalization": _sampled_fisher_normalization,
        "matrix_normalization": _sampled_fisher_score_matrix_normalization,
        "streaming_label": "sampled Fisher",
        "vector_label": "sampled Fisher vector",
        "result_label": "sampled Fisher result",
        "blockwise_label": "sampled_score_gradients",
        "require_streaming": _skip_score_fisher_requirement,
        "require_matrix": _skip_score_fisher_requirement,
        "check_result": _check_sampled_fisher_exact_bound,
        "vmap_error_message": (
            "sampled Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _require_sampled_fisher_semantics,
    },
    "empirical_fisher_vp": {
        "paths": EMPIRICAL_FISHER_VECTOR_VMAP_PATHS,
        "streaming_paths": EMPIRICAL_FISHER_GRADIENT_PRODUCT_PATHS,
        "blockwise_path": EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH,
        "dense_path": EMPIRICAL_FISHER_DENSE_PATH,
        "block_batch_key": "per_example_gradient_blocks",
        "matrix_batch_key": "per_example_gradients",
        "streaming_matrix": _empirical_fisher_gradients,
        "streaming_normalization": _empirical_fisher_streaming_normalization,
        "blockwise_normalization": _empirical_fisher_blockwise_normalization,
        "matrix_normalization": _empirical_fisher_score_matrix_normalization,
        "streaming_label": "empirical Fisher",
        "vector_label": "empirical Fisher vector",
        "result_label": "empirical Fisher result",
        "blockwise_label": "per_example_gradients",
        "require_streaming": _skip_score_fisher_requirement,
        "require_matrix": _skip_score_fisher_requirement,
        "check_result": _skip_score_matrix_result_check,
        "vmap_error_message": (
            "empirical Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _skip_score_fisher_requirement,
    },
}


def _operator_semantic(operator: OperatorSpec, key: str) -> str:
    value = operator.semantics.get(key)

    if not isinstance(value, str):
        message = f"operator semantic field is missing: {key}"
        raise MaterializationError(message)

    return value


def _operator_semantic_int(operator: OperatorSpec, key: str) -> int:
    value = operator.semantics.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        message = f"operator semantic field is missing: {key}"
        raise MaterializationError(message)

    return value


def _operator_semantic_positive_int(operator: OperatorSpec, key: str) -> int:
    value = _operator_semantic_int(operator, key)

    if value < 1:
        message = f"operator semantic field must be positive: {key}"
        raise MaterializationError(message)

    return value
