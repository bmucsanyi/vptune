"""Shared execution substrate for the standard runtime.

Declared-value access, settings validation, finite checks, tensor-tree
runtime operations, and the execution dataclasses used by the runtime
owner modules.
"""

import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, TypeGuard

import torch

from vptune.axes.admission import (
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_call_core_settings,
    admit_forward_ad,
    admit_functional_call,
    admit_torch_func,
)
from vptune.core.data import (
    Batch,
    BufferTree,
    Candidate,
    CandidateOperation,
    FunctionObjective,
    ModuleCallSpec,
    ObjectiveContext,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    ReferenceCheck,
    ScalarObjective,
)
from vptune.core.identities import tensor_signature, to_json_value
from vptune.core.tensor_tree import (
    TensorTree,
    tree_from_leaves,
    tree_leaves,
    tree_map2,
)
from vptune.engine.checks import (
    STANDARD_THRESHOLDS,
    numeric_error_bound_measurements,
    validate_numeric_error_bound,
)
from vptune.errors import (
    AdmissionError,
    ReferenceFailedError,
    RuntimeValueError,
)

MATRIX_FREE_RUNTIME_BINDINGS = ContextVar[
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]] | None
](
    "vptune_matrix_free_runtime_bindings",
    default=None,
)

MIN_SAMPLED_FISHER_FORMULA_BOUND_SAMPLES = 2

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

COMPILE_SETTING_KEYS = (
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

FISHER_SCORE_GRADIENT_PATHS = (
    FISHER_SCORE_GRADIENT_LOOP_PATH,
    FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    FISHER_SCORE_GRADIENT_VMAP_PATH,
    FISHER_BACKWARD_MATERIALIZED_PATH,
)

SAMPLED_FISHER_SCORE_GRADIENT_PATHS = (
    SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
)

EMPIRICAL_FISHER_GRADIENT_PATHS = (
    EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
    EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
    EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
    EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
)

PER_EXAMPLE_GRADIENT_PATHS = (
    PER_EXAMPLE_GRADIENT_LOOP_PATH,
    PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
    PER_EXAMPLE_GRADIENT_BACKWARD_PATH,
    PER_EXAMPLE_GRADIENT_VMAP_PATH,
)

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
    "inverse_metric.preconditioner_product",
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

FISHER_VECTOR_PATH_ROWS = (
    (
        "fisher_vp",
        FISHER_DENSE_PATH,
        FISHER_SCORE_GRADIENT_LOOP_PATH,
        FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        FISHER_SCORE_GRADIENT_VMAP_PATH,
        FISHER_BACKWARD_MATERIALIZED_PATH,
        FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    ),
    (
        "sampled_fisher_vp",
        SAMPLED_FISHER_DENSE_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
        SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
        SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
        SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
    ),
    (
        "empirical_fisher_vp",
        EMPIRICAL_FISHER_DENSE_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
        EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
        EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH,
    ),
)

FISHER_MANUAL_BATCH_PATHS_BY_KIND = {
    family: (loop, torch_func, backward)
    for family, _, loop, torch_func, _, backward, _ in FISHER_VECTOR_PATH_ROWS
}

FISHER_STREAMING_PRODUCT_PATHS_BY_KIND = {
    family: (*FISHER_MANUAL_BATCH_PATHS_BY_KIND[family], vmap)
    for family, _, _, _, vmap, _, _ in FISHER_VECTOR_PATH_ROWS
}

FISHER_VECTOR_VMAP_PATHS_BY_KIND = {
    family: (dense, *FISHER_STREAMING_PRODUCT_PATHS_BY_KIND[family], blockwise)
    for family, dense, _, _, _, _, blockwise in FISHER_VECTOR_PATH_ROWS
}

FISHER_DENSE_PATH_BY_KIND = {
    family: dense for family, dense, _, _, _, _, _ in FISHER_VECTOR_PATH_ROWS
}

FISHER_BLOCKWISE_PATH_BY_KIND = {
    family: blockwise for family, _, _, _, _, _, blockwise in FISHER_VECTOR_PATH_ROWS
}

FISHER_VECTOR_LOOP_PATHS_BY_KIND = {
    family: (dense, loop, torch_func, vmap, backward, blockwise)
    for family, dense, loop, torch_func, vmap, backward, blockwise in (
        FISHER_VECTOR_PATH_ROWS
    )
}

FISHER_MANUAL_PER_EXAMPLE_PATHS = {
    path for paths in FISHER_MANUAL_BATCH_PATHS_BY_KIND.values() for path in paths
}

FISHER_SAMPLE_MANUAL_PER_EXAMPLE_PATHS = {
    path
    for family in ("fisher_vp", "sampled_fisher_vp")
    for path in FISHER_MANUAL_BATCH_PATHS_BY_KIND[family]
}

FISHER_SAMPLE_VMAP_PATHS = {
    vmap
    for family, _, _, _, vmap, _, _ in FISHER_VECTOR_PATH_ROWS
    if family in {"fisher_vp", "sampled_fisher_vp"}
}

VMAP_RUNTIME_PATHS = tuple(vmap for _, _, _, _, vmap, _, _ in FISHER_VECTOR_PATH_ROWS)

EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = FISHER_MANUAL_BATCH_PATHS_BY_KIND[
    "empirical_fisher_vp"
]


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreMatrixCompileRow:
    """ScoreMatrixCompileRow for standard runtime execution."""

    boundary: str
    path_key: str
    paths: tuple[str, ...]
    message: str


SCORE_MATRIX_COMPILE_PATH_VALUES = (
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "vmap_grad",
    "backward_materialized_grad",
)

SCORE_MATRIX_COMPILE_ROWS = {
    "fisher_vp": ScoreMatrixCompileRow(
        boundary="fisher_score_grad",
        path_key="fisher.score_grad_path",
        paths=FISHER_SCORE_GRADIENT_PATHS,
        message="fisher score-gradient boundary requires a score-gradient path",
    ),
    "sampled_fisher_vp": ScoreMatrixCompileRow(
        boundary="sampled_fisher_score_grad",
        path_key="sampled_fisher.score_grad_path",
        paths=SAMPLED_FISHER_SCORE_GRADIENT_PATHS,
        message=(
            "sampled Fisher score-gradient boundary requires a score-gradient path"
        ),
    ),
    "empirical_fisher_vp": ScoreMatrixCompileRow(
        boundary="empirical_fisher_example_grad",
        path_key="empirical_fisher.grad_path",
        paths=EMPIRICAL_FISHER_GRADIENT_PATHS,
        message=("empirical Fisher example-gradient boundary requires a gradient path"),
    ),
    "per_example_gradient": ScoreMatrixCompileRow(
        boundary="per_example_gradient",
        path_key="per_example_gradient.grad_path",
        paths=PER_EXAMPLE_GRADIENT_PATHS,
        message="per-example gradient boundary requires a gradient path",
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

JVP_VECTOR_VMAP_PATHS = (JVP_PATH, JVP_LINEARIZE_PATH)

VJP_VECTOR_VMAP_PATHS = (VJP_PATH,)

GGN_VECTOR_VMAP_PATHS = (
    GGN_JVP_HESSIAN_VJP_PATH,
    GGN_LINEARIZE_HESSIAN_VJP_PATH,
)

VECTOR_VMAP_RUNTIME_PATHS = {
    "jvp": JVP_VECTOR_VMAP_PATHS,
    "vjp": VJP_VECTOR_VMAP_PATHS,
    "hvp": HVP_VECTOR_VMAP_PATHS,
    "ggnvp": GGN_VECTOR_VMAP_PATHS,
    **FISHER_VECTOR_VMAP_PATHS_BY_KIND,
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
    **FISHER_VECTOR_LOOP_PATHS_BY_KIND,
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

CheckpointContextFns = Mapping[str, Callable[[], Any]]

IntermediateTransform = Callable[[TensorTree], TensorTree]


def is_tensor_tree_dict(value: TensorTree) -> TypeGuard[dict[str, TensorTree]]:
    """Return the is tensor tree dict.

    Returns:
        The tensor tree dict.
    """
    return isinstance(value, dict)


def is_tensor_tree_tuple(value: TensorTree) -> TypeGuard[tuple[TensorTree, ...]]:
    """Return the is tensor tree tuple.

    Returns:
        The tensor tree tuple.
    """
    return isinstance(value, tuple)


def direct_operation(
    function: Callable[..., TensorTree],
    args: Sequence[Any],
) -> CandidateOperation:
    """Return the direct operation.

    Returns:
        The operation.
    """

    def operation() -> TensorTree:
        return function(*args)

    return operation


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionChild:
    """Child operator reference used by sequential composition."""

    name: str
    candidate: Candidate
    component: Callable[[Batch, TensorTree], TensorTree]
    anchor_component: Callable[[Batch, TensorTree], TensorTree]
    reference_check: ReferenceCheck
    input_signature: Mapping[str, Any]


def merge_component_measurements(
    measurements: dict[str, Any],
    component_errors: Mapping[str, Mapping[str, float]],
) -> None:
    """Merge component measurements."""
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


def require_min_probe_norm(vector: TensorTree, name: str) -> None:
    """Validate min probe norm.

    Raises:
        ReferenceFailedError: If the declared inputs are invalid.
    """
    norm = float(flatten_vector(vector).norm().detach().cpu())
    min_norm = STANDARD_THRESHOLDS["min_probe_norm"]

    if norm >= min_norm:
        return

    message = f"{name} norm is below min_probe_norm"
    raise ReferenceFailedError(message)


def matrix_symmetry_error(matrix: torch.Tensor) -> float:
    """Return the matrix symmetry error.

    Returns:
        The symmetry error.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise RuntimeValueError(message)

    require_finite_tensor(matrix, "metric matrix")

    return float((matrix - matrix.T).abs().max().item())


def matrix_psd_violation(matrix: torch.Tensor) -> float:
    """Return the matrix psd violation.

    Returns:
        The psd violation.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise RuntimeValueError(message)

    require_finite_tensor(matrix, "metric matrix")

    min_eigenvalue = torch.linalg.eigvalsh(matrix).min()

    return float(torch.clamp(-min_eigenvalue, min=0.0).item())


def matrix_condition_number(matrix: torch.Tensor) -> float:
    """Return the matrix condition number.

    Returns:
        The condition number.
    """
    require_finite_tensor(matrix, "metric matrix")
    condition = torch.linalg.cond(matrix)

    return float(condition.item())


def inverse_residual(
    matrix: torch.Tensor,
    inverse_result: torch.Tensor,
    vector: torch.Tensor,
) -> float:
    """Return the inverse residual.

    Returns:
        The residual.
    """
    require_finite_tensor(matrix, "metric matrix")
    require_finite_tensor(inverse_result, "inverse result")
    require_finite_tensor(vector, "metric vector")

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


def call_with_deferred_finite_checks(
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    """Call with deferred finite checks.

    Returns:
        The with deferred finite checks.
    """
    with deferred_runtime_finite_checks():
        return callback(*args)


@contextlib.contextmanager
def disabled_backend_settings() -> Iterator[None]:
    """Return the disabled backend settings.

    Yields:
        The backend settings items.
    """
    previous = BACKEND_SETTINGS_ENABLED[0]
    BACKEND_SETTINGS_ENABLED[0] = False

    try:
        yield
    finally:
        BACKEND_SETTINGS_ENABLED[0] = previous


def require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate finite tensor.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if not FINITE_CHECKS_ENABLED[0]:
        return

    check_tensor = _finite_check_tensor(tensor)

    if not torch.isfinite(check_tensor).all().item():
        message = f"{name} contains nonfinite values"
        raise RuntimeValueError(message)


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


def require_finite_tree(tree: TensorTree, name: str) -> None:
    """Validate finite tree."""
    for leaf in tree_leaves(tree):
        require_finite_tensor(leaf, name)


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


def tensor_args(value: Any) -> tuple[torch.Tensor, ...]:
    """Return the tensor args.

    Returns:
        The args.
    """
    if isinstance(value, torch.Tensor):
        return (value,)

    if isinstance(value, Mapping):
        return tuple(
            tensor for child in value.values() for tensor in tensor_args(child)
        )

    if isinstance(value, tuple):
        return tuple(tensor for child in value for tensor in tensor_args(child))

    return ()


def require_finite_nested_tensors(value: Any, name: str) -> None:
    """Validate finite nested tensors."""
    for tensor in tensor_args(value):
        require_finite_tensor(tensor, name)


def has_activation_settings(settings: Mapping[str, Any]) -> bool:
    """Return the has activation settings.

    Returns:
        The activation settings.
    """
    return any(key.startswith(("activation.", "checkpoint.")) for key in settings)


def apply_numeric_error_bound(
    measurements: dict[str, float],
    thresholds: Mapping[str, float],
    settings: Mapping[str, Any],
    bound_fields: Mapping[str, Any],
    reference: TensorTree,
) -> None:
    """Apply numeric error bound."""
    bound_measurements = numeric_error_bound_measurements(
        settings,
        bound_fields,
        reference,
    )
    validate_numeric_error_bound(measurements, thresholds, bound_measurements)
    measurements.update(bound_measurements)


def scaled_loss_hessian_batch(batch: Batch, scale: float) -> Batch:
    """Return the scaled loss hessian batch.

    Returns:
        The loss hessian batch.
    """
    if "loss_hessian" not in batch:
        return batch

    result = dict(batch)
    result["loss_hessian"] = _scaled_tensor(
        batch["loss_hessian"],
        scale,
        "loss_hessian",
    )

    return result


def _scaled_tensor(value: Any, scale: float, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        message = f"{name} must be a tensor"
        raise RuntimeValueError(message)

    return value * scale


def dense_jacobian_row_block(
    output: torch.Tensor,
    parameter_leaves: tuple[torch.Tensor, ...],
    parameter_width: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Return the dense jacobian row block.

    Returns:
        The jacobian row block.
    """
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


def class_block_size_with_exact_global_normalization(
    settings: Mapping[str, Any],
) -> int:
    """Return the class block size with exact global normalization.

    Returns:
        The block size with exact global normalization.
    """
    key = "chunk.class_block_size_with_exact_global_normalization"

    return required_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def cotangent_blocks(
    output_cotangent: TensorTree,
    block_size: int,
) -> tuple[TensorTree, ...]:
    """Return the cotangent blocks.

    Returns:
        The blocks.
    """
    flat = flatten_vector(output_cotangent)
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


def dense_jacobian_tree(
    function: Callable[..., torch.Tensor],
    parameter_leaves: tuple[torch.Tensor, ...],
    output_numel: int,
) -> torch.Tensor:
    """Return the dense jacobian tree.

    Returns:
        The jacobian tree.
    """
    jacobian = torch.autograd.functional.jacobian(function, parameter_leaves)
    jacobian_leaves = (jacobian,) if isinstance(jacobian, torch.Tensor) else jacobian
    parts = tuple(
        part.reshape(output_numel, parameter.numel())
        for part, parameter in zip(jacobian_leaves, parameter_leaves, strict=True)
    )

    return torch.cat(parts, dim=1)


def require_loss_hessian_shape(
    loss_hessian: torch.Tensor,
    output_numel: int,
) -> None:
    """Validate loss hessian shape.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if loss_hessian.shape != (output_numel, output_numel):
        message = "loss_hessian shape must match flattened function output"
        raise RuntimeValueError(message)


def parameter_column_ranges(
    width: int,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> tuple[tuple[int, int], ...] | None:
    """Return the parameter column ranges.

    Returns:
        The column ranges.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    block_size = _parameter_block_size(settings)
    layer_block_size = _layer_block_size(settings)

    if block_size is not None and layer_block_size is not None:
        message = "parameter and layer chunk sizes cannot both be set"
        raise RuntimeValueError(message)

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
        raise RuntimeValueError(message)

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
            raise RuntimeValueError(message)

        ranges.append((chunk_start, chunk_stop))

    if ranges[0][0] != 0 or ranges[-1][1] != width:
        message = "chunk.layer_block_size ranges must cover parameter width"
        raise RuntimeValueError(message)

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
            raise RuntimeValueError(message)

        ranges.append((start, stop))

    return tuple(ranges)


def _parameter_block_ranges(
    width: int,
    block_size: int,
) -> Iterator[tuple[int, int]]:
    for start in range(0, width, block_size):
        yield start, min(start + block_size, width)


STREAMING_GRADIENT_LOOP_PATHS = (
    FISHER_SCORE_GRADIENT_LOOP_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
    EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
)

STREAMING_GRADIENT_TORCH_FUNC_PATHS = (
    FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_TORCH_FUNC_PATH,
    EMPIRICAL_FISHER_TORCH_FUNC_GRAD_PATH,
)

STREAMING_GRADIENT_BACKWARD_PATHS = (
    FISHER_BACKWARD_MATERIALIZED_PATH,
    SAMPLED_FISHER_BACKWARD_MATERIALIZED_PATH,
    EMPIRICAL_FISHER_BACKWARD_MATERIALIZED_PATH,
)

STREAMING_GRADIENT_VMAP_PATHS = (
    FISHER_SCORE_GRADIENT_VMAP_PATH,
    SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
)

PER_EXAMPLE_GRADIENT_LOOP_PATHS = (
    *STREAMING_GRADIENT_LOOP_PATHS,
    PER_EXAMPLE_GRADIENT_LOOP_PATH,
)

PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATHS = (
    *STREAMING_GRADIENT_TORCH_FUNC_PATHS,
    PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATH,
)

PER_EXAMPLE_GRADIENT_BACKWARD_PATHS = (
    *STREAMING_GRADIENT_BACKWARD_PATHS,
    PER_EXAMPLE_GRADIENT_BACKWARD_PATH,
)

PER_EXAMPLE_GRADIENT_VMAP_PATHS = (PER_EXAMPLE_GRADIENT_VMAP_PATH,)


def require_nonempty_per_example_terms(terms: torch.Tensor, label: str) -> None:
    """Validate nonempty per example terms.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if terms.numel() == 0:
        message = f"{label} requires at least one objective term"
        raise RuntimeValueError(message)


def per_example_batch_size(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
    label: str,
) -> int:
    """Return the per example batch size.

    Returns:
        The example batch size.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            if dim < 0:
                dim += value.ndim

            return value.shape[dim]

    message = f"{label} requires a nonempty mapped batch"
    raise RuntimeValueError(message)


def per_example_batch_slice(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Return the per example batch slice.

    Returns:
        The example batch slice.
    """
    result = {}

    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            result[key] = value.narrow(dim, start, stop - start)
        else:
            result[key] = value

    return result


def manual_vector_batch_size(settings: Mapping[str, Any]) -> int:
    """Return the manual vector batch size.

    Returns:
        The vector batch size.
    """
    key = "vectorization.batch_size"

    return required_positive_int_setting(
        settings,
        key,
        "vectorization.mode=manual_batch requires positive vectorization.batch_size",
    )


def optional_positive_int_setting(
    settings: Mapping[str, Any],
    key: str,
    invalid_message: str,
) -> int | None:
    """Return the optional positive int setting.

    Returns:
        The positive int setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeValueError(invalid_message)

    return value


def required_positive_int_setting(
    settings: Mapping[str, Any],
    key: str,
    invalid_message: str,
    *,
    missing_message: str | None = None,
) -> int:
    """Return the required positive int setting.

    Returns:
        The positive int setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = optional_positive_int_setting(settings, key, invalid_message)

    if value is None:
        raise RuntimeValueError(
            invalid_message if missing_message is None else missing_message
        )

    return value


def validate_vector_tensor_in_dim(
    vector: torch.Tensor,
    raw_in_dim: Any,
) -> int | None:
    """Validate vector tensor in dim.

    Returns:
        The vector tensor in dim.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if raw_in_dim is None:
        return None

    if not isinstance(raw_in_dim, int) or isinstance(raw_in_dim, bool):
        message = "vectorization.in_dims values must be integers or None"
        raise RuntimeValueError(message)

    dim = raw_in_dim

    if dim < 0:
        dim += vector.ndim

    if dim < 0 or dim >= vector.ndim:
        message = "vectorization.in_dims axis is out of range"
        raise RuntimeValueError(message)

    if vector.shape[dim] == 0:
        message = "vectorized vector inputs require a nonempty mapped dimension"
        raise RuntimeValueError(message)

    return raw_in_dim


def vector_tree_batch_size(vector: TensorTree, in_dims: Any) -> int:
    """Return the vector tree batch size.

    Returns:
        The tree batch size.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    sizes = []
    _collect_vector_tree_batch_sizes(vector, in_dims, sizes)

    if not sizes:
        message = "vectorized vector inputs require at least one mapped leaf"
        raise RuntimeValueError(message)

    first_size = sizes[0]

    for size in sizes[1:]:
        if size != first_size:
            message = "vectorized vector mapped dimensions differ"
            raise RuntimeValueError(message)

    return first_size


def _collect_vector_tree_batch_sizes(
    vector: TensorTree,
    in_dims: Any,
    sizes: list[int],
) -> None:
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return

        dim = normalized_vector_dim(vector, in_dims)
        sizes.append(vector.shape[dim])
        return

    if is_tensor_tree_dict(vector):
        for key in vector:
            _collect_vector_tree_batch_sizes(vector[key], in_dims[key], sizes)

        return

    if is_tensor_tree_tuple(vector):
        for value, in_dim in zip(vector, in_dims, strict=True):
            _collect_vector_tree_batch_sizes(value, in_dim, sizes)

        return

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise RuntimeValueError(message)


def vector_tree_select(vector: TensorTree, in_dims: Any, index: int) -> TensorTree:
    """Return the vector tree select.

    Returns:
        The tree select.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return vector

        dim = normalized_vector_dim(vector, in_dims)

        return vector.select(dim, index)

    if is_tensor_tree_dict(vector):
        return {
            key: vector_tree_select(vector[key], in_dims[key], index) for key in vector
        }

    if is_tensor_tree_tuple(vector):
        return tuple(
            vector_tree_select(value, in_dim, index)
            for value, in_dim in zip(vector, in_dims, strict=True)
        )

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise RuntimeValueError(message)


def vector_tree_slice(
    vector: TensorTree,
    in_dims: Any,
    start: int,
    stop: int,
) -> TensorTree:
    """Return the vector tree slice.

    Returns:
        The tree slice.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if isinstance(vector, torch.Tensor):
        if in_dims is None:
            return vector

        dim = normalized_vector_dim(vector, in_dims)

        return vector.narrow(dim, start, stop - start)

    if is_tensor_tree_dict(vector):
        return {
            key: vector_tree_slice(vector[key], in_dims[key], start, stop)
            for key in vector
        }

    if is_tensor_tree_tuple(vector):
        return tuple(
            vector_tree_slice(value, in_dim, start, stop)
            for value, in_dim in zip(vector, in_dims, strict=True)
        )

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise RuntimeValueError(message)


def normalized_vector_dim(vector: torch.Tensor, in_dim: int) -> int:
    """Return the normalized vector dim.

    Returns:
        The vector dim.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    dim = in_dim

    if dim < 0:
        dim += vector.ndim

    if dim < 0 or dim >= vector.ndim:
        message = "vectorization.in_dims axis is out of range"
        raise RuntimeValueError(message)

    return dim


def stack_tensor_trees(outputs: Sequence[TensorTree], dim: int) -> TensorTree:
    """Stack tensor trees.

    Returns:
        The tensor trees.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if not outputs:
        message = "cannot stack an empty tensor-tree sequence"
        raise RuntimeValueError(message)

    first = outputs[0]

    if isinstance(first, torch.Tensor):
        leaves = []

        for output in outputs:
            if not isinstance(output, torch.Tensor):
                message = "tensor tree structures differ"
                raise RuntimeValueError(message)

            leaves.append(output)

        return torch.stack(tuple(leaves), dim=dim)

    if is_tensor_tree_dict(first):
        return _stack_tensor_tree_dicts(outputs, first, dim)

    if is_tensor_tree_tuple(first):
        return _stack_tensor_tree_tuples(outputs, first, dim)

    message = f"unsupported tensor tree node: {type(first).__name__}"
    raise RuntimeValueError(message)


def cat_tensor_trees(outputs: Sequence[TensorTree], dim: int) -> TensorTree:
    """Return the cat tensor trees.

    Returns:
        The tensor trees.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if not outputs:
        message = "cannot concatenate an empty tensor-tree sequence"
        raise RuntimeValueError(message)

    first = outputs[0]

    if isinstance(first, torch.Tensor):
        leaves = []

        for output in outputs:
            if not isinstance(output, torch.Tensor):
                message = "tensor tree structures differ"
                raise RuntimeValueError(message)

            leaves.append(output)

        return torch.cat(tuple(leaves), dim=dim)

    if is_tensor_tree_dict(first):
        dict_outputs = _require_tensor_tree_dict_outputs(outputs, first)

        return {
            key: cat_tensor_trees(tuple(output[key] for output in dict_outputs), dim)
            for key in first
        }

    if is_tensor_tree_tuple(first):
        tuple_outputs = _require_tensor_tree_tuple_outputs(outputs, first)

        return tuple(
            cat_tensor_trees(tuple(output[index] for output in tuple_outputs), dim)
            for index in range(len(first))
        )

    message = f"unsupported tensor tree node: {type(first).__name__}"
    raise RuntimeValueError(message)


def _stack_tensor_tree_dicts(
    outputs: Sequence[TensorTree],
    first: dict[str, TensorTree],
    dim: int,
) -> TensorTree:
    dict_outputs = _require_tensor_tree_dict_outputs(outputs, first)
    result = {}

    for key in first:
        child_outputs = tuple(output[key] for output in dict_outputs)
        result[key] = stack_tensor_trees(tuple(child_outputs), dim)

    return result


def _require_tensor_tree_dict_outputs(
    outputs: Sequence[TensorTree],
    first: dict[str, TensorTree],
) -> tuple[dict[str, TensorTree], ...]:
    result = []

    for output in outputs:
        if not is_tensor_tree_dict(output) or set(output) != set(first):
            message = "tensor tree mapping keys differ"
            raise RuntimeValueError(message)

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
        result.append(stack_tensor_trees(tuple(child_outputs), dim))

    return tuple(result)


def _require_tensor_tree_tuple_outputs(
    outputs: Sequence[TensorTree],
    first: tuple[TensorTree, ...],
) -> tuple[tuple[TensorTree, ...], ...]:
    result = []

    for output in outputs:
        if not is_tensor_tree_tuple(output) or len(output) != len(first):
            message = "tensor tree sequence lengths differ"
            raise RuntimeValueError(message)

        result.append(output)

    return tuple(result)


def runtime_path_from_settings(
    settings: Mapping[str, Any],
    operator_kind: str,
    setting_key: str,
) -> str:
    """Return the runtime path from settings.

    Returns:
        The path from settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get(setting_key)

    if not isinstance(value, str):
        message = f"{setting_key} is required"
        raise RuntimeValueError(message)

    path = SPEC_PATH_TO_RUNTIME[operator_kind].get(value)

    if path is None:
        message = f"{setting_key} value is not lowered: {value}"
        raise RuntimeValueError(message)

    return path


def require_positive_definite_matrix(matrix: torch.Tensor, label: str) -> None:
    """Validate positive definite matrix.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    _, info = torch.linalg.cholesky_ex(matrix)

    if torch.any(info != 0):
        message = f"{label} requires positive definite matrix"
        raise RuntimeValueError(message)


MetricMultiplyRunner = Callable[
    [OperatorSpec, Batch, TensorTree, Mapping[str, Any]],
    TensorTree,
]


def zero_numerator_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Return the zero numerator divide.

    Returns:
        The numerator divide.
    """
    return torch.where(
        numerator == 0,
        torch.zeros_like(numerator),
        numerator / denominator,
    )


def batched_cg_residual_satisfies_tolerance(
    residual: torch.Tensor,
    right_hand_sides: torch.Tensor,
    tolerance: float | None,
) -> bool:
    """Return the batched cg residual satisfies tolerance.

    Returns:
        The cg residual satisfies tolerance.
    """
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


def map_flat_batch(
    flat_batch: torch.Tensor,
    runner: Callable[[torch.Tensor], torch.Tensor],
    label: str,
) -> torch.Tensor:
    """Return the map flat batch.

    Returns:
        The flat batch.
    """
    result = torch.stack(
        tuple(runner(flat_vector) for flat_vector in flat_batch),
        dim=0,
    )
    require_finite_tensor(result, label)

    return result


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricBlock:
    """One Kronecker-factored metric block for a matrix parameter."""

    parameter_name: str
    left_factor_key: str
    right_factor_key: str


def batch_signature(value: Any) -> Any:
    """Return the batch signature.

    Returns:
        The signature.
    """
    if isinstance(value, torch.Tensor):
        return tensor_signature(value)

    if isinstance(value, Mapping):
        return {
            str(key): batch_signature(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(batch_signature(item) for item in value)

    return to_json_value(value)


def run_with_matrix_free_runtime_bindings(
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    callback: Callable[[], Any],
) -> Any:
    """Run with matrix free runtime bindings.

    Returns:
        The with matrix free runtime bindings.
    """
    token = MATRIX_FREE_RUNTIME_BINDINGS.set(dict(bindings))

    try:
        return callback()
    finally:
        MATRIX_FREE_RUNTIME_BINDINGS.reset(token)


def require_transform_admission_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate transform admission settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if _requires_torch_func_admission(operator, path, settings):
        try:
            admit_torch_func(settings)
        except AdmissionError as error:
            raise RuntimeValueError(str(error)) from error

    if path in {
        JVP_PATH,
        JVP_FORWARD_AD_PATH,
        HVP_JVP_GRAD_PATH,
        GGN_JVP_HESSIAN_VJP_PATH,
    }:
        try:
            admit_forward_ad(settings)
        except AdmissionError as error:
            raise RuntimeValueError(str(error)) from error


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


def supports_vector_loop(operator_kind: str, path: str | None) -> bool:
    """Return the supports vector loop.

    Returns:
        The vector loop.
    """
    return path in VECTOR_LOOP_RUNTIME_PATHS.get(operator_kind, ())


def require_call_runtime_settings(settings: Mapping[str, Any]) -> None:
    """Validate call runtime settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if any(key in settings for key in FUNCTIONAL_CALL_FIELDS):
        try:
            admit_functional_call(settings)
        except AdmissionError as error:
            raise RuntimeValueError(str(error)) from error

    path = _require_call_path_settings(settings)
    _require_call_state_settings(settings, path)
    _require_call_return_settings(settings, path)


def _require_call_path_settings(settings: Mapping[str, Any]) -> Any:
    try:
        return admit_call_core_settings(settings)
    except AdmissionError as error:
        raise RuntimeValueError(str(error)) from error


def _require_call_state_settings(
    settings: Mapping[str, Any],
    path: Any,
) -> None:
    tied_weights = settings.get("call.tied_weights")

    if tied_weights not in {None, "preserve_alias_groups"}:
        message = f"call.tied_weights is unsupported: {tied_weights}"
        raise RuntimeValueError(message)

    parametrizations = settings.get("call.parametrizations")

    if parametrizations not in {None, "preserve_parametrizations"}:
        message = f"call.parametrizations is unsupported: {parametrizations}"
        raise RuntimeValueError(message)

    buffer_mutation = settings.get("call.buffer_mutation")

    if buffer_mutation == "declared_and_restored":
        if path == "stateful_module":
            _require_stateful_declared_restore_settings(settings)
        else:
            _require_declared_state_restore_settings(settings)
    elif buffer_mutation not in {None, "forbidden"}:
        message = f"call.buffer_mutation is unsupported: {buffer_mutation}"
        raise RuntimeValueError(message)


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
        raise RuntimeValueError(message)


def _require_declared_state_restore_settings(settings: Mapping[str, Any]) -> None:
    if (
        settings.get("call.path") != "functional_call"
        or settings.get("call.params") != "explicit_params"
        or settings.get("call.buffers") != "explicit_buffers"
    ):
        message = "declared state restoration requires explicit functional-call inputs"
        raise RuntimeValueError(message)

    try:
        admit_functional_call(settings)
    except AdmissionError as error:
        raise RuntimeValueError(str(error)) from error

    if settings["mutates_state"] is not True:
        message = "declared state restoration requires mutates_state=True"
        raise RuntimeValueError(message)


def _require_stateful_declared_restore_settings(settings: Mapping[str, Any]) -> None:
    try:
        admit_functional_call(settings)
    except AdmissionError as error:
        raise RuntimeValueError(str(error)) from error

    if settings["mutates_state"] is not True:
        message = "declared state restoration requires mutates_state=True"
        raise RuntimeValueError(message)


def require_stateful_module_path_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate stateful module path settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
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
    raise RuntimeValueError(message)


def require_module_state_names(
    module: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
) -> None:
    """Validate module state names.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    module_params = dict(module.named_parameters())
    module_buffers = dict(module.named_buffers())

    if set(params) != set(module_params):
        message = "module parameter names must match runtime params"
        raise RuntimeValueError(message)

    if set(buffers) != set(module_buffers):
        message = "module buffer names must match runtime buffers"
        raise RuntimeValueError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class _ModuleStateSlot:
    parent: torch.nn.Module
    name: str
    kind: str
    value: torch.Tensor | None


def replace_module_state(
    module: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
) -> tuple[_ModuleStateSlot, ...]:
    """Replace module state.

    Returns:
        The module state.
    """
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


def restore_module_state(slots: tuple[_ModuleStateSlot, ...]) -> None:
    """Restore module state.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    for slot in reversed(slots):
        if slot.kind == "parameter":
            slot.parent.__dict__["_parameters"][slot.name] = slot.value
        elif slot.kind == "buffer":
            slot.parent.__dict__["_buffers"][slot.name] = slot.value
        else:
            message = f"module state slot kind is unsupported: {slot.kind}"
            raise RuntimeValueError(message)


def _module_state_parent(
    module: torch.nn.Module,
    key: str,
) -> tuple[torch.nn.Module, str]:
    parent_name, separator, state_name = key.rpartition(".")
    parent = module.get_submodule(parent_name) if separator else module

    return parent, state_name


def invoke_stateful_module(
    module: torch.nn.Module,
    call: ModuleCallSpec,
    batch: Batch,
) -> object:
    """Invoke stateful module.

    Returns:
        The stateful module.
    """
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
    raise RuntimeValueError(message)


def select_stateful_module_output(
    output: object,
    call: ModuleCallSpec,
    settings: Mapping[str, Any],
) -> object:
    """Select stateful module output.

    Returns:
        The stateful module output.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    return_type = settings.get("call.return_type")

    if return_type in {None, "raw_tensor_tree"}:
        if call.output_fields:
            message = "raw_tensor_tree module calls must not declare output fields"
            raise RuntimeValueError(message)

        return output

    if return_type != "model_output_object_with_declared_fields":
        message = f"call.return_type is unsupported: {return_type}"
        raise RuntimeValueError(message)

    if not call.output_fields:
        message = "model output object rows require declared output fields"
        raise RuntimeValueError(message)

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
    raise RuntimeValueError(message)


def _select_indexed_output_field(value: object, index: int) -> object:
    if isinstance(value, (tuple, list)):
        try:
            return value[index]
        except IndexError as error:
            message = f"model output index is missing: {index}"
            raise RuntimeValueError(message) from error

    message = f"model output is not indexable at: {index}"
    raise RuntimeValueError(message)


def require_batch_data_axis(operator: OperatorSpec, label: str) -> None:
    """Validate batch data axis.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if operator.data_axis == "batch":
        return

    message = f"{label} requires operator data_axis=batch"
    raise RuntimeValueError(message)


def require_per_token_schedule(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    batch_layout_callback: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    """Validate per token schedule.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    block_size = token_block_size(settings)
    schedule_value = settings.get("schedule.per_token")

    if schedule_value is None:
        if block_size is not None:
            message = "chunk.token_block_size requires schedule.per_token=loop"
            raise RuntimeValueError(message)

        return

    if schedule_value == "loop":
        if block_size is not None:
            _require_token_block_runtime(operator, settings)

        return

    if schedule_value == "packed":
        if settings.get("input.batch_layout") not in {
            "packed_with_inverse_permutation",
            "variable_length",
        }:
            message = "schedule.per_token=packed requires packed input binding"
            raise RuntimeValueError(message)

        if batch_layout_callback is None:
            message = "schedule.per_token=packed requires packed input binding"
            raise RuntimeValueError(message)

        return

    message = f"schedule.per_token is unsupported: {schedule_value}"
    raise RuntimeValueError(message)


def token_block_size(settings: Mapping[str, Any]) -> int | None:
    """Return the token block size.

    Returns:
        The block size.
    """
    key = "chunk.token_block_size"
    return optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


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
    raise RuntimeValueError(message)


def require_lm_head_chunking_settings(
    settings: Mapping[str, Any],
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    """Validate lm head chunking settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    key = "chunk.lm_head_weight_chunk_bytes"
    value = optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )

    if value is None:
        return

    if lm_head_chunker is None:
        message = f"{key} requires LM-head weight binding"
        raise RuntimeValueError(message)


def require_per_example_schedule(path: str | None, value: Any) -> None:
    """Validate per example schedule.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if value == "loop":
        if path in FISHER_MANUAL_PER_EXAMPLE_PATHS:
            return

        message = f"schedule.per_example=loop is incompatible with path: {path}"
        raise RuntimeValueError(message)

    if value == "vmap":
        if path in VMAP_RUNTIME_PATHS:
            return

        message = f"schedule.per_example=vmap is incompatible with path: {path}"
        raise RuntimeValueError(message)

    if value == "manual_batch":
        if path in FISHER_MANUAL_PER_EXAMPLE_PATHS:
            return

        message = f"schedule.per_example=manual_batch is incompatible with path: {path}"
        raise RuntimeValueError(message)

    message = f"schedule.per_example is unsupported: {value}"
    raise RuntimeValueError(message)


def require_per_example_batch_size_setting(
    path: str | None,
    settings: Mapping[str, Any],
    key: str,
    allowed_paths: set[str],
    manual_paths: set[str],
) -> None:
    """Validate per example batch size setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    schedule = settings.get("schedule.per_example")

    if key not in settings:
        if schedule == "manual_batch" and path in manual_paths:
            message = f"{key} is required for schedule.per_example=manual_batch"
            raise RuntimeValueError(message)

        return

    if not (
        (schedule == "vmap" and path in allowed_paths)
        or (schedule == "manual_batch" and path in manual_paths)
    ):
        message = (
            f"{key} requires schedule.per_example=vmap or manual_batch on a "
            "matching path"
        )
        raise RuntimeValueError(message)

    required_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def require_output_buffer_settings(settings: Mapping[str, Any]) -> None:
    """Validate output buffer settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get("memory.output_buffers")

    if value is None or value == "fresh_allocation":
        return

    if value == "preallocated":
        return

    message = f"memory.output_buffers is unsupported: {value}"
    raise RuntimeValueError(message)


def require_fusion_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    fusion_rewriter: Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None,
) -> None:
    """Validate fusion settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    for key, fused_values in _fusion_value_domains().items():
        value = settings.get(key)

        if value is None or value == "model_default":
            continue

        if value in fused_values:
            if key == "fusion.loss":
                _require_fused_loss_identity(operator, value)

            if fusion_rewriter is None:
                message = f"{key}={value} requires a registered fused implementation"
                raise RuntimeValueError(message)

            continue

        message = f"{key} is unsupported: {value}"
        raise RuntimeValueError(message)


def _fusion_value_domains() -> Mapping[str, set[str]]:
    return {
        "fusion.norm": {"fused_rmsnorm", "fused_layernorm"},
        "fusion.mlp": {"fused_mlp"},
        "fusion.rope": {"fused_rope"},
        "fusion.logits": {"fused_logits_projection"},
        "fusion.loss": {"fused_ce", "fused_kl"},
    }


def _require_fused_loss_identity(
    operator: OperatorSpec,
    fused_loss: Any,
) -> None:
    expected_loss_kind = {
        "fused_ce": "softmax_cross_entropy",
        "fused_kl": "kl",
    }.get(fused_loss)

    if expected_loss_kind is None:
        return

    loss = operator.semantics.get("loss")

    if not isinstance(loss, Mapping):
        message = (
            f"fusion.loss={fused_loss} requires typed "
            f"{expected_loss_kind} loss identity"
        )
        raise RuntimeValueError(message)

    if loss.get("kind") != expected_loss_kind:
        message = (
            f"fusion.loss={fused_loss} requires typed "
            f"{expected_loss_kind} loss identity"
        )
        raise RuntimeValueError(message)

    identity = loss.get("identity")

    if not isinstance(identity, Mapping):
        message = f"fusion.loss={fused_loss} requires exact global normalization fields"
        raise RuntimeValueError(message)

    reduction = identity.get("reduction")
    denominator = identity.get("denominator")

    if not isinstance(reduction, str) or not isinstance(denominator, str):
        message = f"fusion.loss={fused_loss} requires exact global normalization fields"
        raise RuntimeValueError(message)


def runtime_fusion_module(
    module: torch.nn.Module | None,
    candidate: Candidate,
    fusion_rewriter: Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None,
) -> torch.nn.Module | None:
    """Return the runtime fusion module.

    Returns:
        The fusion module.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if not _has_fused_setting(candidate.settings):
        return module

    if module is None:
        message = "fused rows require a module"
        raise RuntimeValueError(message)

    if fusion_rewriter is None:
        message = "fused rows require a registered fused implementation"
        raise RuntimeValueError(message)

    return fusion_rewriter(module, candidate)


def _has_fused_setting(settings: Mapping[str, Any]) -> bool:
    for key, fused_values in _fusion_value_domains().items():
        if settings.get(key) in fused_values:
            return True

    return False


def require_path_coupled_reuse_setting(
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
    """Validate path coupled reuse setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if operator.kind != operator_kind:
        return

    value = settings.get(setting_key)

    if value is None or value == default_value:
        return

    if value != required_value:
        message = f"{setting_key} is unsupported: {value}"
        raise RuntimeValueError(message)

    if path != required_path:
        raise RuntimeValueError(path_message)


def require_output_cotangent_block_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate output cotangent block settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    block_size = output_cotangent_block_size(settings)

    if block_size is None:
        return

    if operator.kind != "ggnvp":
        message = "chunk.output_cotangent_block_size applies only to GGNVP rows"
        raise RuntimeValueError(message)

    if path == GGN_DENSE_PATH:
        message = "chunk.output_cotangent_block_size requires a GGN VJP path"
        raise RuntimeValueError(message)


def output_cotangent_block_size(settings: Mapping[str, Any]) -> int | None:
    """Return the output cotangent block size.

    Returns:
        The cotangent block size.
    """
    key = "chunk.output_cotangent_block_size"
    return optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def _parameter_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.parameter_block_size"
    return optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def _layer_block_size(settings: Mapping[str, Any]) -> int | None:
    key = "chunk.layer_block_size"
    return optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def require_parameter_block_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> None:
    """Validate parameter block size settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    block_size = _parameter_block_size(settings)
    layer_block_size = _layer_block_size(settings)

    if block_size is None and layer_block_size is None:
        return

    if block_size is not None and layer_block_size is not None:
        message = "parameter and layer chunk sizes cannot both be set"
        raise RuntimeValueError(message)

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
    raise RuntimeValueError(message)


def _parameter_surface_width(
    parameter_surface: ParameterSurface | None,
    key: str,
) -> int:
    if parameter_surface is None:
        message = f"{key} requires declared layer_groups"
        raise RuntimeValueError(message)

    return sum(math.prod(shape) for shape in parameter_surface.shapes)


def declared_parameter_groups(
    parameter_surface: ParameterSurface | None,
    group_field: str,
    key: str,
) -> tuple[tuple[str, ...], ...]:
    """Return the declared parameter groups.

    Returns:
        The parameter groups.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if parameter_surface is None:
        message = f"{key} requires declared {group_field}"
        raise RuntimeValueError(message)

    groups = (
        parameter_surface.layer_groups
        if group_field == "layer_groups"
        else parameter_surface.block_groups
    )

    if not groups:
        message = f"{key} requires declared {group_field}"
        raise RuntimeValueError(message)

    return groups


def wrap_grouped_parameter_tree(
    tree: ParameterTree,
    groups: tuple[tuple[str, ...], ...],
    key: str,
) -> ParameterTree:
    """Wrap grouped parameter tree.

    Returns:
        The grouped parameter tree.
    """
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
        raise RuntimeValueError(message)


def parameter_tree_from_tensor_tree(
    tree: TensorTree,
    label: str,
) -> ParameterTree:
    """Return the parameter tree from tensor tree.

    Returns:
        The tree from tensor tree.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if not isinstance(tree, dict):
        message = f"{label} requires a parameter-tree tensor mapping"
        raise RuntimeValueError(message)

    result = dict[str, torch.Tensor]()

    for key, value in tree.items():
        if not isinstance(key, str):
            message = f"{label} requires string keys"
            raise RuntimeValueError(message)

        if not isinstance(value, torch.Tensor):
            message = f"{label} requires tensor leaves"
            raise RuntimeValueError(message)

        result[key] = value

    return result


def pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the pin cpu tensor.

    Returns:
        The cpu tensor.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if tensor.device.type != "cpu":
        message = "teacher output pinning requires CPU tensors"
        raise RuntimeValueError(message)

    try:
        return tensor.pin_memory()
    except RuntimeError as error:
        raise RuntimeValueError(str(error)) from error


def runtime_named_tensor_map_preserve_alias(
    tree: dict[str, torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the runtime named tensor map preserve alias.

    Returns:
        The named tensor map preserve alias.
    """
    mapped = {}
    result = {}

    for key, tensor in tree.items():
        alias_key = id(tensor)

        if alias_key not in mapped:
            mapped[alias_key] = function(tensor)

        result[key] = mapped[alias_key]

    return result


def require_parameter_surface_runtime_settings(
    parameter_surface: ParameterSurface | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate parameter surface runtime settings.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if parameter_surface is None:
        return

    if (
        preserves_parameter_aliases(settings)
        and parameter_surface.tied_weights_policy != "preserve"
    ):
        message = "tied-weight preservation requires preserved parameter surface"
        raise RuntimeValueError(message)


def preserves_parameter_aliases(settings: Mapping[str, Any]) -> bool:
    """Return the preserves parameter aliases.

    Returns:
        The parameter aliases.
    """
    return (
        settings.get("call.tied_weights") == "preserve_alias_groups"
        or settings.get("layout.aliasing") == "preserve_tied_weight_aliases"
    )


def has_parameter_aliases(params: ParameterTree) -> bool:
    """Return the has parameter aliases.

    Returns:
        The parameter aliases.
    """
    ids = tuple(id(tensor) for tensor in params.values())

    return len(ids) != len(set(ids))


def runtime_output_to_buffer(
    output: TensorTree,
    buffer: TensorTree | None,
) -> TensorTree:
    """Return the runtime output to buffer.

    Returns:
        The output to buffer.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if buffer is None:
        return output

    try:
        return tree_map2(_copy_output_tensor, buffer, output)
    except (RuntimeError, TypeError) as error:
        message = "memory.output_buffers=preallocated output tree mismatch"
        raise RuntimeValueError(message) from error


def _copy_output_tensor(buffer: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if buffer.dtype != output.dtype:
        message = "preallocated output buffer dtype mismatch"
        raise RuntimeError(message)

    if buffer.device != output.device:
        message = "preallocated output buffer device mismatch"
        raise RuntimeError(message)

    buffer.copy_(output)

    return buffer


def runtime_batch_value(value: Any, dtype: torch.dtype) -> Any:
    """Return the runtime batch value.

    Returns:
        The batch value.
    """

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(dtype=dtype)

        return tensor

    return runtime_nested_tensor_value(value, convert)


def runtime_nested_tensor_value(
    value: Any,
    map_tensor: Callable[[torch.Tensor], Any],
    *,
    error_message: str | None = None,
) -> Any:
    """Return the runtime nested tensor value.

    Returns:
        The nested tensor value.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if isinstance(value, torch.Tensor):
        return map_tensor(value)

    if isinstance(value, dict):
        return {
            key: runtime_nested_tensor_value(
                child,
                map_tensor,
                error_message=error_message,
            )
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            runtime_nested_tensor_value(
                child,
                map_tensor,
                error_message=error_message,
            )
            for child in value
        )

    if error_message is not None:
        raise RuntimeValueError(error_message)

    return value


def run_with_call_grad_mode(
    settings: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    """Run with call grad mode.

    Returns:
        The with call grad mode.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get("call.grad_mode")

    if value is None:
        return callback()

    if value == "grad_enabled":
        with torch.enable_grad():
            return callback()

    message = f"call.grad_mode is unsupported: {value}"
    raise RuntimeValueError(message)


def declared_tensor_snapshot(
    values: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    label: str,
) -> dict[str, torch.Tensor]:
    """Return the declared tensor snapshot.

    Returns:
        The tensor snapshot.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    snapshot = {}

    for key in keys:
        if key not in values:
            message = f"declared mutated {label} is missing: {key}"
            raise RuntimeValueError(message)

        snapshot[key] = values[key].detach().clone()

    return snapshot


def restore_declared_tensors(
    values: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
) -> None:
    """Restore declared tensors."""
    with torch.no_grad():
        for key, tensor in snapshot.items():
            values[key].copy_(tensor)


def buffer_snapshot(buffers: BufferTree) -> BufferTree:
    """Return the buffer snapshot.

    Returns:
        The snapshot.
    """
    return {key: tensor.detach().clone() for key, tensor in buffers.items()}


def require_buffers_unchanged(before: BufferTree, after: BufferTree) -> None:
    """Validate buffers unchanged.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if set(before) != set(after):
        message = "call.buffer_mutation=forbidden detected changed buffer keys"
        raise RuntimeValueError(message)

    for key, before_tensor in before.items():
        after_tensor = after[key]

        if before_tensor.shape != after_tensor.shape:
            message = f"call.buffer_mutation=forbidden changed buffer shape: {key}"
            raise RuntimeValueError(message)

        if before_tensor.dtype != after_tensor.dtype:
            message = f"call.buffer_mutation=forbidden changed buffer dtype: {key}"
            raise RuntimeValueError(message)

        if before_tensor.device != after_tensor.device:
            message = f"call.buffer_mutation=forbidden changed buffer device: {key}"
            raise RuntimeValueError(message)

        if not torch.equal(before_tensor, after_tensor):
            message = f"call.buffer_mutation=forbidden changed buffer value: {key}"
            raise RuntimeValueError(message)


def autocast_setting(settings: Mapping[str, Any]) -> tuple[str, torch.dtype] | None:
    """Return the autocast setting.

    Returns:
        The setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get("autocast")

    if value is None or value == "off":
        return None

    if value == "cuda_fp16":
        return _cuda_autocast(torch.float16)

    if value == "cuda_bf16":
        return _cuda_autocast(torch.bfloat16)

    message = f"autocast is unsupported by standard runtime: {value}"
    raise RuntimeValueError(message)


def _cuda_autocast(dtype: torch.dtype) -> tuple[str, torch.dtype]:
    if not torch.cuda.is_available():
        message = "CUDA autocast requires CUDA"
        raise RuntimeValueError(message)

    return "cuda", dtype


def bool_string_setting(settings: Mapping[str, Any], key: str) -> bool | None:
    """Return the bool string setting.

    Returns:
        The string setting.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = settings.get(key)

    if value is None:
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be true or false"
    raise RuntimeValueError(message)


def spec_path_value_for_runtime_path(operator_kind: str, path: str) -> str | None:
    """Return the spec path value for runtime path.

    Returns:
        The path value for runtime path.
    """
    path_map = SPEC_PATH_TO_RUNTIME.get(operator_kind, {})

    for spec_value, runtime_path in path_map.items():
        if runtime_path == path:
            return spec_value

    return None


def require_candidate_family(operator: OperatorSpec, candidate: Candidate) -> None:
    """Validate candidate family.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if candidate.family != operator.family:
        message = (
            f"candidate family does not match operator family: "
            f"{candidate.family} != {operator.family}"
        )
        raise RuntimeValueError(message)


def require_path(
    operator_kind: str,
    path: str,
    allowed_paths: tuple[str, ...],
) -> None:
    """Validate path.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if path in allowed_paths:
        return

    message = f"operator path {path} does not support {operator_kind}"
    raise RuntimeValueError(message)


def scalar_objective(
    operator: OperatorSpec,
    scalar_objectives: Mapping[str, ScalarObjective],
) -> ScalarObjective:
    """Return the scalar objective.

    Returns:
        The objective.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    objective = scalar_objectives.get(operator.objective_id)

    if objective is None:
        message = f"scalar objective is missing: {operator.objective_id}"
        raise RuntimeValueError(message)

    return objective


def function_objective(
    operator: OperatorSpec,
    function_objectives: Mapping[str, FunctionObjective],
) -> FunctionObjective:
    """Return the function objective.

    Returns:
        The objective.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    objective = function_objectives.get(operator.objective_id)

    if objective is None:
        message = f"function objective is missing: {operator.objective_id}"
        raise RuntimeValueError(message)

    return objective


def checked_function_output(
    settings: Mapping[str, Any],
    output: object,
    name: str,
) -> TensorTree:
    """Return the checked function output.

    Returns:
        The function output.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if settings.get("call.return_type") != "raw_tensor_tree":
        if not _is_raw_tensor_tree(output):
            message = f"{name} must be a tensor tree"
            raise RuntimeValueError(message)

        return output

    if not _is_raw_tensor_tree(output):
        message = f"{name} must be a raw tensor tree"
        raise RuntimeValueError(message)

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


def grad_enabled_params(params: ParameterTree) -> ParameterTree:
    """Return the grad enabled params.

    Returns:
        The enabled params.
    """
    return {
        name: tensor.detach().requires_grad_(True) for name, tensor in params.items()
    }


def parameter_grad_tree(params: ParameterTree) -> TensorTree:
    """Return the parameter grad tree.

    Returns:
        The grad tree.
    """
    return tree_from_leaves(
        params,
        tuple(
            torch.zeros_like(param) if param.grad is None else param.grad.detach()
            for param in params.values()
        ),
    )


def flatten_vector(vector: TensorTree) -> torch.Tensor:
    """Flatten vector.

    Returns:
        The vector.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    leaves = tree_leaves(vector)

    if not leaves:
        message = "dense standard operator requires at least one tensor leaf"
        raise RuntimeValueError(message)

    return torch.cat(tuple(leaf.reshape(-1) for leaf in leaves))


def flatten_vector_like(template: TensorTree, vector: TensorTree) -> torch.Tensor:
    """Flatten vector like.

    Returns:
        The vector like.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if isinstance(vector, torch.Tensor):
        return vector.reshape(-1)

    if isinstance(template, torch.Tensor):
        message = "flat tensor template requires a flat tensor vector"
        raise RuntimeValueError(message)

    leaves = matching_vector_leaves(template, vector)

    return torch.cat(tuple(leaf.reshape(-1) for leaf in leaves))


def matching_vector_leaves(
    params: TensorTree, vector: TensorTree
) -> tuple[torch.Tensor, ...]:
    """Return the matching vector leaves.

    Returns:
        The vector leaves.
    """
    checked = tree_map2(
        lambda param, tangent: tangent.reshape_as(param), params, vector
    )

    return tree_leaves(checked)


def wrap_flat_parameter_tree(
    template: ParameterTree,
    result: torch.Tensor,
) -> ParameterTree:
    """Wrap flat parameter tree.

    Returns:
        The flat parameter tree.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    leaves = []
    offset = 0

    for leaf in template.values():
        width = leaf.numel()
        leaves.append(result[offset : offset + width].reshape_as(leaf))
        offset += width

    if offset != result.numel():
        message = "flat parameter layout length differs from parameter tree"
        raise RuntimeValueError(message)

    return dict(zip(template, leaves, strict=True))


def flatten_vector_batch(
    template: TensorTree,
    vector: TensorTree,
    in_dims: Any,
) -> torch.Tensor:
    """Flatten vector batch.

    Returns:
        The vector batch.
    """
    batch_size = vector_tree_batch_size(vector, in_dims)
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

    if is_tensor_tree_dict(template) and is_tensor_tree_dict(vector):
        if not isinstance(in_dims, Mapping) or set(in_dims) != set(template):
            message = "vectorization.in_dims must match the vector tree"
            raise RuntimeValueError(message)

        if set(vector) != set(template):
            message = "vector tree mapping keys differ from parameters"
            raise RuntimeValueError(message)

        for key in template:
            _collect_flat_vector_batch_pieces(
                template[key],
                vector[key],
                in_dims[key],
                batch_size,
                pieces,
            )

        return

    if is_tensor_tree_tuple(template) and is_tensor_tree_tuple(vector):
        if not isinstance(in_dims, tuple) or len(in_dims) != len(template):
            message = "vectorization.in_dims must match the vector tree"
            raise RuntimeValueError(message)

        if len(vector) != len(template):
            message = "vector tree sequence length differs from parameters"
            raise RuntimeValueError(message)

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
    raise RuntimeValueError(message)


def _flat_vector_batch_piece(
    template: torch.Tensor,
    vector: torch.Tensor,
    in_dim: Any,
    batch_size: int,
) -> torch.Tensor:
    if in_dim is None:
        if vector.shape != template.shape:
            message = "unmapped vector leaf shape differs from parameter leaf"
            raise RuntimeValueError(message)

        return vector.reshape(1, -1).expand(batch_size, -1)

    if not isinstance(in_dim, int) or isinstance(in_dim, bool):
        message = "vectorization.in_dims values must be integers or None"
        raise RuntimeValueError(message)

    dim = normalized_vector_dim(vector, in_dim)
    unbatched_shape = vector.shape[:dim] + vector.shape[dim + 1 :]

    if unbatched_shape != template.shape:
        message = "mapped vector leaf shape differs from parameter leaf"
        raise RuntimeValueError(message)

    if vector.shape[dim] != batch_size:
        message = "vectorized vector mapped dimensions differ"
        raise RuntimeValueError(message)

    return vector.movedim(dim, 0).reshape(batch_size, -1)


def wrap_flat_vector(template: TensorTree, result: torch.Tensor) -> TensorTree:
    """Wrap flat vector.

    Returns:
        The flat vector.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    leaves = []
    offset = 0

    for leaf in tree_leaves(template):
        width = leaf.numel()
        leaves.append(result[offset : offset + width].reshape_as(leaf))
        offset += width

    if offset != result.numel():
        message = "dense standard operator output length differs from vector tree"
        raise RuntimeValueError(message)

    return tree_from_leaves(template, tuple(leaves))


def wrap_flat_vector_batch(template: TensorTree, result: torch.Tensor) -> TensorTree:
    """Wrap flat vector batch.

    Returns:
        The flat vector batch.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    if result.ndim != MATRIX_DIMS:
        message = "batched flat vector result must be a matrix"
        raise RuntimeValueError(message)

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
        raise RuntimeValueError(message)

    return tree_from_leaves(template, tuple(leaves))


def batch_tensor(batch: Batch, key: str) -> torch.Tensor:
    """Return the batch tensor.

    Returns:
        The tensor.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        message = f"batch tensor is missing: {key}"
        raise RuntimeValueError(message)

    return value


def batch_tensor_blocks(batch: Batch, key: str) -> tuple[torch.Tensor, ...]:
    """Return the batch tensor blocks.

    Returns:
        The tensor blocks.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = batch.get(key)

    if not isinstance(value, tuple) or not value:
        message = f"batch tensor blocks are missing: {key}"
        raise RuntimeValueError(message)

    row_count = None

    for block in value:
        if not isinstance(block, torch.Tensor):
            message = f"batch tensor block must be a tensor: {key}"
            raise RuntimeValueError(message)

        if block.ndim != MATRIX_DIMS:
            message = f"batch tensor block must be two-dimensional: {key}"
            raise RuntimeValueError(message)

        if block.shape[0] == 0:
            message = f"batch tensor blocks must have at least one row: {key}"
            raise RuntimeValueError(message)

        require_finite_tensor(block, key)

        if row_count is None:
            row_count = block.shape[0]
        elif block.shape[0] != row_count:
            message = f"batch tensor blocks must share row count: {key}"
            raise RuntimeValueError(message)

    return value


def batch_tree(batch: Batch, key: str) -> TensorTree:
    """Return the batch tree.

    Returns:
        The tree.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
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
    raise RuntimeValueError(message)


def normalization(batch: Batch, operator: OperatorSpec) -> float:
    """Return the normalization.

    Returns:
        The normalization.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = batch.get("normalization")

    if not isinstance(value, int | float):
        message = "batch normalization is missing"
        raise RuntimeValueError(message)

    normalization = float(value)

    if normalization <= 0.0:
        message = "batch normalization must be positive"
        raise RuntimeValueError(message)

    if operator.aggregation == "sum" and not math.isclose(
        normalization,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        message = "sum aggregation requires normalization=1.0"
        raise RuntimeValueError(message)

    return normalization


def sampling_bound_float(bound: Mapping[str, Any], key: str) -> float:
    """Return the sampling bound float.

    Returns:
        The bound float.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = bound.get(key)

    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"sampled Fisher sampling_bound.{key} must be numeric"
        raise RuntimeValueError(message)

    result = float(value)

    if not math.isfinite(result) or result < 0.0:
        message = f"sampled Fisher sampling_bound.{key} must be finite and nonnegative"
        raise RuntimeValueError(message)

    return result


def sampling_bound_probability(bound: Mapping[str, Any], key: str) -> float:
    """Return the sampling bound probability.

    Returns:
        The bound probability.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    result = sampling_bound_float(bound, key)

    if 0.0 < result < 1.0:
        return result

    message = f"sampled Fisher sampling_bound.{key} must be in (0, 1)"
    raise RuntimeValueError(message)


def operator_semantic(operator: OperatorSpec, key: str) -> str:
    """Return the operator semantic.

    Returns:
        The semantic.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = operator.semantics.get(key)

    if not isinstance(value, str):
        message = f"operator semantic field is missing: {key}"
        raise RuntimeValueError(message)

    return value


def _operator_semantic_int(operator: OperatorSpec, key: str) -> int:
    value = operator.semantics.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        message = f"operator semantic field is missing: {key}"
        raise RuntimeValueError(message)

    return value


def operator_semantic_positive_int(operator: OperatorSpec, key: str) -> int:
    """Return the operator semantic positive int.

    Returns:
        The semantic positive int.

    Raises:
        RuntimeValueError: If the declared inputs are invalid.
    """
    value = _operator_semantic_int(operator, key)

    if value < 1:
        message = f"operator semantic field must be positive: {key}"
        raise RuntimeValueError(message)

    return value


ActivationUnpackHooks = Mapping[str, Callable[[Any], torch.Tensor]]


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
    compiled_vector_step: Callable[[TensorTree], TensorTree] | None = None
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
