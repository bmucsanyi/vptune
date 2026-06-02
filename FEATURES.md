# vptune Feature Sweep Space

`vptune` tunes executable implementations of matrix-free autodiff operators. A
candidate row is valid only when every setting changes code that actually runs,
communication that actually happens, memory that is actually moved, or layout
that actually affects the measured operator while preserving the declared
operator.

## Fixed Problem Fields

These fields define the problem. They are recorded in input identity and are not
sweep axes:

- model identity and weights
- parameter subset
- buffers
- data rows
- input masks
- vector and cotangent values
- train or eval mode
- RNG semantics
- scalar loss for gradient and HVP
- function output surface for JVP and VJP
- loss and output surface for GGNVP
- score or log-prob definition for FisherVP
- per-example loss definition for empirical FisherVP
- metric definition and damping
- target devices and process count

Rows may change representations of fixed fields only when the row computes the
same operator and passes package-owned reference checks. Sequence packing,
padding changes, attention packing, teacher-output caching, and distributed
placement all fall under this rule.

Reference checks, numeric thresholds, numeric error bounds, and
memory-stability checks are acceptance rules. They are not speed knobs.

## Axis Ownership

Every setting key has exactly one owner. Duplicate owners are invalid.

- Operator-owned axes define the mathematical lowering for one operator family.
- Shared axes define implementation choices used by many operators.
- Applicability lists say which operators may use a shared axis.
- A candidate row contains one flat setting map, but each key is validated by one
  owner.

The package must reject rows that set a key outside its owner or set two keys
that encode the same decision.

## Supported Operator Families

The package sweep space covers:

- gradient
- JVP
- VJP
- HVP
- GGNVP
- FisherVP
- sampled FisherVP
- empirical FisherVP
- metric multiply
- inverse metric multiply
- composition of selected operators

## Operator-Owned Axes

### Gradient

For a scalar objective $f(\theta)$, compute $\nabla_\theta f$.

Owned axes:

- `gradient.path`: `torch_autograd_grad`, `torch_func_grad`, `torch_func_grad_and_value`, `backward_materialized_grad`
- `gradient.value_reuse`: `gradient_only`, `gradient_and_primal_value`
- `gradient.graph_schedule`: `build_once`, `rebuild_per_call`

Shared axes that apply:

- model call
- attention execution
- gradient materialization
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- direct autograd anchor
- finite-difference directional check on small inputs
- segmentation invariance when microbatching or token blocking is used
- full-size agreement for non-math or shape-dependent attention, compile,
  fusion, or sharded-reduction rows

### JVP

For a function $g(\theta)$ and tangent $v$, compute $J_g v$.

Owned axes:

- `jvp.path`: `torch_func_jvp`, `forward_ad_dual`, `torch_func_linearize`
- `jvp.linearize_reuse`: `none`, `reuse_at_same_primal`

Shared axes that apply:

- vectorization
- model call
- attention execution
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Admission requirements:

- `torch.func` rows require transform-compatible pure functions.
- `forward_ad_dual` rows require forward AD coverage for every operation on the
  path.
- `jvp.linearize_reuse=reuse_at_same_primal` requires identical params,
  buffers, inputs, masks, dtype mode, and output surface across reused vectors.

Mandatory checks:

- `torch.func.jvp` or forward AD reference
- finite-difference directional output check on small inputs
- VJP dot identity when a matching VJP is available
- full-size agreement for non-math or shape-dependent attention, compile,
  fusion, or sharded-reduction rows

### VJP

For a function $g(\theta)$ and cotangent $u$, compute $J_g^\top u$.

Owned axes:

- `vjp.path`: `torch_func_vjp`, `autograd_grad_outputs`, `backward_materialized_grad`
- `vjp.closure_reuse`: `none`, `reuse_vjp_closure_at_same_primal`

Shared axes that apply:

- vectorization
- model call
- attention execution
- gradient materialization
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- `torch.func.vjp` or eager autograd reference
- JVP/VJP dot identity: $\langle Jv,u\rangle = \langle v,J^\top u\rangle$
- segmentation invariance for cotangent chunking
- full-size agreement for non-math or shape-dependent attention, compile,
  fusion, or sharded-reduction rows

### HVP

For a scalar objective $f(\theta)$ and vector $v$, compute $H_f v$.

Owned axes:

- `hvp.path`: `reverse_over_reverse`, `jvp_grad`, `autograd_functional_hvp`, `autograd_functional_vhp`, `forward_ad_dual`, `linearize_grad`
- `hvp.graph_schedule`: `retain_graph_across_vectors`, `rebuild_graph_per_vector`
- `hvp.primal_reuse`: `reuse_primal`, `recompute_primal`
- `hvp.gradient_reuse`: `reuse_gradient_closure`, `recompute_gradient`

Shared axes that apply:

- vectorization
- model call
- attention execution
- gradient materialization
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- reverse-over-reverse anchor
- `torch.autograd.functional.hvp` or `vhp` anchor on small inputs
- HVP symmetry: $\langle x,Hy\rangle = \langle y,Hx\rangle$
- finite-difference gradient-direction check
- segmentation invariance for chunked rows
- full-size agreement for non-math or shape-dependent attention, compile,
  fusion, or sharded-reduction rows

### GGNVP

For model output $z(\theta)$, declared loss $\ell(z)$, Jacobian $J$, and vector
$v$, compute $Gv = J^\top H_\ell Jv$.

The loss and output surface define the operator. Cross entropy, KL, retain KL,
token masking, and reductions are fixed problem fields.

Owned axes:

- `ggn.jvp_path`: `torch_func_jvp`, `forward_ad_dual`, `torch_func_linearize`
- `ggn.loss_hessian_path`: `closed_form_softmax_ce_kl`, `autodiff_loss_hvp`
- `ggn.loss_hessian_kernel`: `dense_global`, `streaming_global`, `two_pass_chunked_global`
- `ggn.vjp_path`: `torch_func_vjp`, `autograd_grad_outputs`
- `ggn.jvp_reuse`: `reuse_jvp`, `recompute_jvp`
- `ggn.cotangent_reuse`: `reuse_output_cotangent`, `recompute_output_cotangent`

`ggn.loss_hessian_kernel` owns the softmax/logsumexp implementation for CE and
KL. It is an implementation row only when it changes the measured output-space
Hessian-vector product while preserving exact global normalization.

Exact categorical Fisher for NLL is represented by the CE/KL GGNVP lowering.

Shared axes that apply:

- vectorization
- model call
- attention execution
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny $J^\top H_\ell Jv$
- JVP/VJP dot identity
- PSD check when $H_\ell$ is declared PSD
- segmentation invariance for output chunking
- attention-backend equality on small inputs for attention rows
- full-size agreement for non-math or shape-dependent attention, compile, fusion, or
  sharded-reduction rows

### FisherVP

For declared score gradients $g_i = \nabla_\theta \log p_\theta(y_i \mid x_i)$,
compute $Fv = E[g(g^\top v)]$ over the declared score source.

Owned axes:

- `fisher.expectation_path`: `explicit_full_expectation_score_rows`
- `fisher.score_grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`
- `fisher.accumulation`: `streaming_dot_accumulate`, `materialize_score_gradients`, `blockwise_score_matrix`

Operator fields such as label source, denominator, and score definition are fixed
semantics. Rows may sweep only equivalent implementations.

Shared axes that apply:

- vectorization
- model call
- attention execution
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- explicit score-gradient outer product on small inputs
- dense tiny Fisher matrix multiply
- attention-backend equality for attention rows
- full-size agreement for non-math or shape-dependent attention, compile, fusion, or
  sharded-reduction rows

### Sampled FisherVP

For a fixed sample table $\{y_{is}\}_{s=1}^S$ or a fixed seed and sample count
$S$, compute the sampled Fisher estimator:

$$\hat F_S v = \frac{1}{nS}\sum_{i,s} g_{is}(g_{is}^\top v)$$

Sampled FisherVP is a separate operator family. It computes a stochastic
estimator with a fixed sample table or fixed seed and sample count. Exact
FisherVP comparison is an explicit acceptance check with a declared sampling
bound for the same input signature.

Owned axes:

- `sampled_fisher.sample_source`: `fixed_sample_table`, `fixed_seed_and_count`
- `sampled_fisher.score_grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`
- `sampled_fisher.accumulation`: `streaming_dot_accumulate`, `materialize_score_gradients`, `blockwise_score_matrix`
- `sampled_fisher.exact_fisher_check`: `disabled`, `enabled_with_sampling_bound`

Fixed problem fields:

- sample count $S$
- sample seed or sample table identity
- sampling distribution
- sampling-bound formula used when exact-Fisher comparison is enabled

Shared axes that apply:

- vectorization
- model call
- attention execution
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- fixed-seed repeatability or sample-table equality
- explicit sampled score-gradient outer product on small inputs
- exact-Fisher agreement within the declared sampling bound when
  `sampled_fisher.exact_fisher_check=enabled_with_sampling_bound`
- full-size agreement for non-math or shape-dependent attention, compile, fusion, or
  sharded-reduction rows

### Empirical FisherVP

For per-example gradients $g_i = \nabla_\theta \ell_i(\theta)$, compute
$F_{\mathrm{emp}}v = \frac{1}{n}\sum_i g_i(g_i^\top v)$ with the declared
denominator.

Owned axes:

- `empirical_fisher.grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`
- `empirical_fisher.accumulation`: `streaming_dot_accumulate`, `materialize_per_example_gradients`, `blockwise_gradient_matrix`

Shared axes that apply:

- vectorization
- model call
- attention execution
- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution

Mandatory checks:

- explicit per-example gradient loop
- dense tiny empirical Fisher matrix multiply
- segmentation invariance
- attention-backend equality for attention rows
- full-size agreement for non-math or shape-dependent attention, compile, fusion, or
  sharded-reduction rows

### Metric Multiply

A metric operator is declared by its mathematical representation. Dense, KFAC,
diagonal, block diagonal, low rank, and GGN-derived metrics are different metric
specs unless the spec declares equivalence.

Owned axes for a fixed metric spec:

- `metric.multiply_path`: `dense_matmul`, `factorized_multiply`, `blockwise_multiply`, `streaming_multiply`
- `metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`
- `metric.accumulation`: `streaming`, `materialized_blocks`

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny metric multiply
- symmetry check when the metric is declared symmetric
- PSD check when the metric is declared PSD

### Inverse Metric Multiply

For a declared metric $M$, compute $M^{-1}v$ or the declared damped inverse.
Damping and the accepted residual define the inverse metric spec.

Owned axes:

- `inverse_metric.solve_path`: `dense_solve`, `cholesky_solve`, `eigh_solve`, `svd_solve`, `conjugate_gradient`, `factorized_solve`, `blockwise_solve`, `woodbury_low_rank_solve`
- `inverse_metric.preconditioner`: `none`, `diagonal`, `block_diagonal`, `factorized_metric`
- `inverse_metric.iteration_budget`: declared positive integer set
- `inverse_metric.factor_reuse`: `refactor_each_rhs`, `reuse_factor_across_rhs`
- `inverse_metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`

`inverse_metric.iteration_budget` is an implementation cap. A row with too low a
cap fails the fixed inverse residual acceptance check. Residual tolerance is not
a sweep axis.

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny solve
- inverse residual against the declared operator: $\|(M+\lambda I)x-v\| / \|v\|$ for damped inverse rows and $\|Mx-v\| / \|v\|$ for undamped inverse rows
- symmetry check when applicable

### Composition

For declared child operators $A_1,\ldots,A_k$, compute the declared composition.
The mathematical order is fixed by the operator spec.

Owned axes:

- `composition.execution`: `materialize_each_child`, `stream_child_outputs`, `fuse_adjacent_children`, `compile_whole_composition`
- `composition.child_evaluation`: `selected_child_rows`, `inline_child_lowering`
- `composition.validation`: `validate_each_child`, `validate_composed_output`

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- child reference checks
- dense tiny composed output when dimensions allow it
- dependency identity equality by direct fields

## Shared Axes

### Vectorization

Applies to JVP, VJP, HVP, GGNVP, FisherVP, sampled FisherVP, empirical
FisherVP, and any composition child that accepts multiple vectors or cotangents.
This axis owns the vector, tangent, or cotangent dimension only. It does not own
data-example batching, per-example gradient batching, token packing, or
microbatching.

- `vectorization.mode`: `single_loop`, `manual_batch`, `vmap`
- `vectorization.batch_size`
- `vectorization.vmap_chunk_size`
- `vectorization.in_dims`
- `vectorization.randomness`: `error`, `same`, `different`

`vmap` rows require transform-compatible code and explicit randomness behavior.

### Gradient Materialization

Applies to gradient, VJP, and HVP rows that compute gradients with respect to the
declared parameter surface.

- `grad_materialization.mode`: `return_tensor_tree`, `materialize_grad_then_read`

`grad_materialization.mode=materialize_grad_then_read` owns the decision to
populate `.grad` and read it back. Clearing stale `.grad` values before probes is
mandatory execution hygiene and is never a row.

Rows whose AD path uses `torch.func` must set
`grad_materialization.mode=return_tensor_tree`.

`gradient.path=backward_materialized_grad` and
`vjp.path=backward_materialized_grad` require
`grad_materialization.mode=materialize_grad_then_read`.

### Model Call And Functionalization

Applies to gradient, JVP, VJP, HVP, GGNVP, FisherVP, sampled FisherVP, and
empirical FisherVP.

- `call.path`: `functional_call`, `stateful_module`
- `call.params`: `explicit_params`, `module_params`
- `call.buffers`: `explicit_buffers`, `module_buffers`
- `call.tied_weights`: `preserve_alias_groups`
- `call.parametrizations`: `preserve_parametrizations`
- `call.buffer_mutation`: `forbidden`, `declared_and_restored`
- `call.grad_mode`: `grad_enabled`
- `call.return_type`: `raw_tensor_tree`, `model_output_object_with_declared_fields`

`torch.func` transform rows must use pure functions. Rows that call
`torch.autograd.grad` inside a transformed function are invalid because PyTorch
documents that this combination is outside the transform model.

### Attention Execution

Applies to operators whose model forward executes attention.

Attention rows distinguish frontend dispatch from kernel dispatch.

- `attention.frontend`: `transformers_eager`, `transformers_sdpa`, `transformers_flash_attention_2`, `transformers_flash_attention_3`, `transformers_flash_attention_4`, `transformers_flex_attention`, `paged|eager`, `paged|sdpa`, `paged|flash_attention_2`, `paged|flash_attention_3`, `paged|flash_attention_4`, `registered_transformers_attention`, `pytorch_sdpa_direct`, `patched_eager`, `packed_exact`, `blockwise_exact`
- `attention.sdpa_kernel`: `math`, `flash_attention`, `efficient_attention`, `cudnn_attention`, `overrideable`, `priority_list`
- `attention.custom_kernel_id`: registered attention implementation id
- `attention.mask_formatter_id`: registered mask formatter id
- `attention.partition`: `full`, `packed_tokens`, `blockwise_queries`, `segmented_forward_ad`
- `attention.padding`: `dense_padded`, `unpadded_packed`

`attention.sdpa_kernel` applies only when the executable calls PyTorch SDPA.
Auto selection is represented by `priority_list` with the exact backend order
recorded in the row.

Fixed attention semantics that every row records and checks:

- causal policy
- sliding-window policy
- padding policy
- mask convention
- dropout probability and RNG semantics
- GQA/MQA head counts and repeat policy
- QKV tensor layout
- head layout
- scale source
- `use_cache`
- `output_attentions`
- RoPE parameters
- position-id policy
- attention logit softcap
- final logit softcap

`output_attentions=True` is fixed by the requested output surface. It is a speed
row only when the requested output includes attention weights.

Backend sweeps require dropout probability zero unless the row declares and
proves identical RNG behavior across backends.

### Batching, Chunking, And Input Representation

Applies to all operators, with family-specific fields admitted only for the
families named in the key.

- `batch.data_microbatch_size`
- `batch.hvp_row_batch_size`
- `batch.ggn_batch_size`
- `batch.fisher_sample_batch_size`
- `batch.empirical_example_batch_size`
- `chunk.token_block_size`
- `chunk.sequence_position_block_size`
- `chunk.class_block_size_with_exact_global_normalization`
- `chunk.output_cotangent_block_size`
- `chunk.parameter_block_size`
- `chunk.layer_block_size`
- `chunk.lm_head_weight_chunk_bytes`
- `schedule.per_example`: `loop`, `vmap`, `manual_batch`
- `schedule.per_token`: `loop`, `packed`
- `schedule.gradient_accumulation`: `single_step`, `microbatch_accumulate`
- `input.batch_layout`: `dense_padded`, `packed_with_inverse_permutation`, `variable_length`
- `input.length_grouping`: `none`, `exact_length_bucket`
- `input.host_to_device`: `outside_measured_call`, `inside_measured_call`
- `input.residency`: `cpu_staged`, `cpu_pinned`, `gpu`
- `teacher_outputs`: `precomputed_cpu`, `precomputed_cpu_pinned`, `precomputed_gpu`, `recomputed_with_equality_check`

Rows may change input representation only when they preserve the same examples,
masks, and logical token order. Class/logit blocking for CE or KL requires exact
global normalization.

`schedule.per_example` owns batching across data examples, including the
per-example-gradient loop or `vmap` used by empirical FisherVP. It does not own
the vector, tangent, or cotangent dimension.

`batch.data_microbatch_size` owns the data split size.
`schedule.gradient_accumulation` selects whether split data chunks are
accumulated. The number of accumulation chunks is derived from the logical batch
size and `batch.data_microbatch_size`.

Packing has three separate facets and invalid combinations are rejected:

- `input.batch_layout=packed_with_inverse_permutation` owns the physical input
  representation and inverse order restoration.
- `schedule.per_token=packed` owns token-loop scheduling over that packed
  representation.
- `attention.partition=packed_tokens` owns the attention implementation on
  packed tokens.

If `schedule.per_token=packed` or `attention.partition=packed_tokens`, then
`input.batch_layout` must be `packed_with_inverse_permutation` or
`variable_length`. If `input.batch_layout=dense_padded`, packed token scheduling
and packed attention are invalid.

### Activation And Memory Schedule

Applies to all operators that allocate intermediates.

- `checkpoint.use_reentrant`: `false`
- `checkpoint.early_stop`: `false`, `true`
- `checkpoint.preserve_rng_state`: `false`, `true`, under fixed RNG semantics
- `checkpoint.determinism_check`: `default`, `none`
- `checkpoint.context_fn`: `none`, `declared_context_pair`
- `memory.primal_outputs`: `retain`, `recompute`
- `memory.jvp_outputs`: `retain`, `recompute`
- `memory.output_cotangents`: `retain`, `recompute`
- `activation.recompute`: `none`, `checkpoint_non_reentrant_by_layer`, `checkpoint_selective`, `manual_recompute`
- `activation.offload`: `none`, `saved_tensor_hooks_cpu`, `custom_saved_tensor_hooks`
- `memory.vector_residency`: `gpu`, `cpu_pinned`, `cpu_staged`, `mmap_cpu`
- `memory.intermediate_residency`: `gpu`, `cpu_pinned`, `cpu_staged`
- `memory.factor_residency`: `gpu`, `cpu_pinned`, `cpu_staged`, `mmap_cpu`
- `memory.output_buffers`: `fresh_allocation`, `preallocated`

Clearing stale gradients between probes is mandatory execution hygiene and is
never a row.

`activation.recompute` owns activation recomputation. `checkpoint.*` owns
checkpoint API details for rows whose recompute mechanism starts with
`checkpoint_`. If `activation.recompute` is not checkpoint-backed, then
`checkpoint.early_stop=false`, `checkpoint.preserve_rng_state=false`,
`checkpoint.determinism_check=none`, and `checkpoint.context_fn=none`.

`activation.offload` owns activation offload through saved-tensor hooks. A row
that uses custom saved-tensor hooks must set
`activation.offload=custom_saved_tensor_hooks`; a row that uses CPU hooks must
set `activation.offload=saved_tensor_hooks_cpu`.

Activation recompute and activation offload are jointly coupled. Rows may
combine them only when the executable applies both mechanisms and the reference
check proves the same operator.

Rows that recompute a region containing RNG-consuming operations must set
`checkpoint.preserve_rng_state=true`. Rows with active recompute must set
`checkpoint.determinism_check=default`.

### Dtype And Numeric Backend

Applies to all operators. Factor dtype for metrics is owned here, not by the
metric sections.

- `dtype.parameter_storage`: `fp32`, `bf16`, `fp16`, `fp8_when_supported`
- `dtype.model_compute`: `fp32`, `bf16`, `fp16`, `fp8_when_supported`
- `dtype.autodiff_compute`: `fp32`, `bf16`, `fp16`
- `dtype.accumulation`: `fp32`, `bf16`, `fp16`
- `dtype.vector`: `fp32`, `bf16`, `fp16`
- `dtype.intermediate`: `fp32`, `bf16`, `fp16`
- `dtype.output`: `fp32`, `bf16`, `fp16`
- `dtype.metric_factor`: `fp32`, `bf16`, `fp16`
- `autocast`: `off`, `cuda_fp16`, `cuda_bf16`
- `numeric.tf32`: `false`, `true`
- `numeric.float32_matmul_precision`: `highest`, `high`, `medium`
- `numeric.bf16_reduced_precision_reduction`: `false`, `true`
- `numeric.fp16_reduced_precision_reduction`: `false`, `true`
- `numeric.deterministic_algorithms`: `false`, `true`
- `numeric.loss_scaling`: `none`, `static_scale_with_exact_unscale`

Rows that degrade reduction precision must pass a derived numeric error bound.
The bound is computed from dtype, reduction length, operation count, and input
scale, then compared to the fixed operator acceptance limit. The row fails when
its measured error exceeds the derived bound or when the derived bound exceeds
the fixed operator acceptance limit.

For a reduction-degrading row, the operator spec records $k$, $\epsilon$,
$C_{\mathrm{op}}$, $S_{\mathrm{row}}$, and the output norm floor. The derived
absolute bound is $B_{\mathrm{abs}} = C_{\mathrm{op}}\gamma_k(\epsilon)S_{\mathrm{row}}$ with $\gamma_k(\epsilon)=k\epsilon/(1-k\epsilon)$ and $k\epsilon < 1$. The derived relative bound is $B_{\mathrm{rel}} = B_{\mathrm{abs}}/\max(\|y_{\mathrm{ref}}\|,\mathrm{floor})$. A row whose bound fields are missing fails admission.

These fields are reduction-degrading rows and always require derived bounds:

- `dtype.accumulation=bf16`
- `dtype.accumulation=fp16`
- `numeric.tf32=true`
- `numeric.float32_matmul_precision=medium`
- `numeric.bf16_reduced_precision_reduction=true`
- `numeric.fp16_reduced_precision_reduction=true`
- `fsdp.mp_policy.reduce_dtype=bf16`
- `fsdp.mp_policy.reduce_dtype=fp16`

`numeric.loss_scaling=static_scale_with_exact_unscale` must state the unscale
law for the operator. Gradient, JVP, VJP, HVP, GGNVP, metric multiply, inverse
metric multiply, and composition rows are degree one in the vector-product
output. Explicit Fisher, sampled Fisher, and empirical Fisher rows using scaled
score gradients are degree two in the score-gradient scale.

### Torch Compile

Applies to every operator with a callable boundary. `torch.compile` is a
first-class sweep axis, and rows must name the callable boundary that is
compiled.

- `compile.enabled`: `false`, `true`
- `compile.boundary`: `model_forward`, `transformer_block`, `attention_module`, `loss_closure`, `gradient_closure`, `jvp_closure`, `vjp_closure`, `hvp_single_vector`, `hvp_batched_vectors`, `ggn_jvp`, `ggn_loss_hessian_product`, `ggn_vjp`, `ggn_full_product`, `fisher_score_grad`, `sampled_fisher_score_grad`, `empirical_fisher_example_grad`, `metric_multiply`, `inverse_metric_solve`, `composition_child`, `whole_operator`
- `compile.backend`: `inductor`, or a registered backend returned by `torch.compiler.list_backends()` that does not own CUDA graph capture
- `compile.mode`: `None`, `default`, `max-autotune`
- `compile.fullgraph`: `false`, `true`
- `compile.dynamic`: `None`, `false`, `true`
- `compile.compiled_autograd`: `false`, `true`
- `compile.options.epilogue_fusion`: `false`, `true`
- `compile.options.shape_padding`: `false`, `true`
- `compile.cuda_graphs`: `false`, `true`
- `compile.cache_state`: `cold_compile`, `warm_cache`

Fields from the PyTorch API that are not sweep axes:

- `name`: record-only identity
- `disable`: represented by `compile.enabled=false`
- option maps that request max autotune: represented by
  `compile.mode=max-autotune`
- backend aliases, mode presets, or backend option maps that request CUDA graph
  capture: represented by `compile.cuda_graphs`
- `mode="reduce-overhead"`: represented by `compile.mode=default` and
  `compile.cuda_graphs=true`
- `mode="max-autotune-no-cudagraphs"`: represented by
  `compile.mode=max-autotune` and `compile.cuda_graphs=false`
- debug options such as tracing and graph diagrams: diagnostics only
- unsafe guard filtering: excluded from default package rows

Compiled rows record compile time, first-call time, steady-state time, number of
recompiles, graph-break status, CUDA graph capture status, peak memory during
compile, and peak memory during steady-state calls.

Selection uses the declared call horizon $N$ and the measured recompile count
$R$: $T_{\mathrm{row}} = ((1+R)T_{\mathrm{compile}} / N) + T_{\mathrm{steady}}$.

### Kernel Fusion

Applies to model-call operators when the fused kernel supports the needed AD
order.

- `fusion.norm`: `model_default`, `fused_rmsnorm`, `fused_layernorm`
- `fusion.mlp`: `model_default`, `fused_mlp`
- `fusion.rope`: `model_default`, `fused_rope`
- `fusion.logits`: `model_default`, `fused_logits_projection`
- `fusion.loss`: `model_default`, `fused_ce`, `fused_kl`

Each fused row must pass the operator reference checks for the selected AD path.
For HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, and any
composition containing those operators, fused rows must also pass a
double-backward or equivalent higher-order agreement check at a
kernel-triggering size. `fusion.loss=fused_ce` and `fusion.loss=fused_kl`
require exact global normalization.

### Parameter And Vector Layout

Applies to all operators. DTensor placement is owned by distributed execution;
layout values that use DTensor must reference the corresponding `dtensor.*`
placement setting.

- `layout.params`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`
- `layout.vector`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`
- `layout.output`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`
- `layout.flatten_order`: `canonical_parameter_order`
- `layout.vector_ops`: `python_loop`, `foreach`
- `layout.contiguity`: `contiguous`, `preserve_existing_strides`
- `layout.aliasing`: `preserve_tied_weight_aliases`
- `layout.parametrizations`: `preserve_active_parametrizations`

### Distributed Execution

Applies to rows that run under a declared process group and measure all ranks.
DTensor placement and distributed output placement are owned here.

- `distributed.launch`: `single_process`, `torchrun`
- `distributed.process_group_backend`: `nccl`, `gloo`, `ucc_when_available`
- `distributed.local_rank_binding`: `cuda_local_rank`, `explicit_device_map`
- `distributed.mesh_shape`
- `distributed.mesh_dim_names`
- `distributed.strategy`: `single_gpu`, `fsdp2`, `hsdp`, `tensor_parallel`, `sequence_parallel`, `context_parallel`, `hybrid`
- `dtensor.params_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.vector_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.logits_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.tangent_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.cotangent_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.output_placement`: `replicate`, `shard_dim`, `partial`
- `dtensor.redistribute_schedule`: `none`, `before_forward`, `before_backward`, `between_operator_parts`, `before_output`

FSDP2 rows:

- `fsdp.wrap_granularity`: `root`, `transformer_block`, `block_group`
- `fsdp.reshard_after_forward`: `false`, `true`, integer group size when supported
- `fsdp.shard_placement_fn`: `none`, `declared_fn`
- `fsdp.mp_policy.param_dtype`: `fp32`, `bf16`, `fp16`
- `fsdp.mp_policy.reduce_dtype`: `fp32`, `bf16`, `fp16`
- `fsdp.mp_policy.output_dtype`: `fp32`, `bf16`, `fp16`
- `fsdp.mp_policy.cast_forward_inputs`: `false`, `true`
- `fsdp.offload_policy`: `none`, `cpu`
- `fsdp.ignored_params`: declared parameter set
- `fsdp.dp_mesh_dims`: declared mesh dimensions

Tensor-parallel rows:

- `tp.plan`: registered model plan id
- `tp.qkv_projection`: `colwise`, `rowwise`, `replicated`
- `tp.output_projection`: `rowwise`, `colwise`, `replicated`
- `tp.mlp_up_gate`: `colwise`, `rowwise`, `replicated`
- `tp.mlp_down`: `rowwise`, `colwise`, `replicated`
- `tp.embedding`: `replicated`, `rowwise`, `colwise`
- `tp.lm_head`: `replicated`, `vocab_sharded`
- `tp.prepare_module_input`: declared input layout conversion
- `tp.prepare_module_output`: declared output layout conversion
- `tp.loss_parallel`: `false`, `true`

`tp.loss_parallel=true` requires exact cross-shard normalization for CE or KL
losses and a multi-rank agreement check against the unsharded loss on the same
logical batch.

Sequence and context rows:

- `sequence_parallel.enabled`: `false`, `true`
- `sequence_parallel.norm_modules`: declared module names
- `sequence_parallel.output_placement_policy`: `preserve_sequence_shard`, `redistribute_to_declared_output`
- `context_parallel.enabled`: `false`, `true`
- `context_parallel.rotate_method`: `all_gather`, `all_to_all`
- `context_parallel.sequence_dim`

Communication rows:

- `comm.overlap`: `none`, `all_gather_overlap`, `reduce_scatter_overlap`, `both`
- `comm.prefetch`: `none`, `forward`, `backward`, `both`
- `comm.collective_bucket_size`
- `comm.rank_memory_reduction`: `max_peak_allocated`, `max_peak_reserved`, `sum_peak_reserved`

Distributed selection records elapsed wall time after rank barriers, rank-local
memory, global memory reductions, and rank failure sets.

## Search Space Factoring

The full cross product is too large for real LLMs. Search must use axis ownership
and coupling classes.

### Integer Domains

Every integer-valued axis must have a finite domain before candidate generation.
The domain is either an explicit value set or an `AutobatchDomain`.

Explicit value-set fields:

- `values`: positive integer tuple
- `owner`: axis owner id
- `admission`: backend and memory precheck identity

`AutobatchDomain` fields:

- `min_value`
- `max_value`
- `initial_value`
- `growth`: `doubling`, `linear_step`, `declared_sequence`
- `objective`: `largest_passing`, `fastest_passing`
- `failure_signals`: OOM, reference failure, runtime failure, backend rejection
- `termination`: exhausted declared values or bracketed failure frontier
- `settings_for_value`: mapping from integer value to concrete candidate settings

`admission` cannot build a candidate table until every integer axis has one of
these finite domains.

### Axis Coupling Classes

Class A and Class B are search annotations on keys that still belong to one
Class C primary group. They do not remove keys from the Class C partition. Class
C is the complete partition used to form merged search groups.

Class A, mostly independent after admission:

- `layout.vector_ops`
- `gradient.value_reuse`
- `grad_materialization.mode`

These can be swept separately inside a fixed operator path, dtype, attention,
layout, and distributed setting.

Class B, conditionally independent:

- input residency and host-to-device staging
- compile cache state
- fixed teacher-output residency

These can be swept separately only after the row fixes the operator path,
attention frontend, compile state, and distributed strategy.

Class C primary groups form a partition:

- `ad_lowering`: `gradient.*`, `jvp.*`, `vjp.*`, `hvp.*`, `ggn.*`,
  `fisher.*`, `sampled_fisher.*`, `empirical_fisher.*`, `composition.*`,
  `grad_materialization.mode`, `vectorization.*`, and `call.*`
- `attention_dispatch`: `attention.frontend`, `attention.sdpa_kernel`,
  `attention.custom_kernel_id`, `attention.mask_formatter_id`,
  `attention.partition`, and `attention.padding`
- `input_schedule`: `batch.*`, `chunk.*`, `schedule.*`, `input.*`, and
  `teacher_outputs`
- `activation_memory`: `checkpoint.*`, `activation.*`,
  `memory.primal_outputs`, `memory.jvp_outputs`, `memory.output_cotangents`,
  `memory.vector_residency`, and `memory.intermediate_residency`
- `numeric_backend`: `dtype.*`, `autocast`, and `numeric.*`
- `distributed_layout`: `layout.*`, `dtensor.*`, `distributed.*`, `fsdp.*`,
  `tp.*`, `sequence_parallel.*`, `context_parallel.*`, and `comm.*`
- `compile`: `compile.*` and `memory.output_buffers`
- `fusion`: `fusion.*`
- `metric_storage`: `metric.*` and `memory.factor_residency`
- `inverse_solve`: `inverse_metric.*`

Class C merge rules:

- If `attention.partition=packed_tokens`, merge `attention_dispatch` with
  `input_schedule`.
- If `compile.boundary=attention_module`, merge `compile` with
  `attention_dispatch`.
- If `compile.boundary` names an operator part, merge `compile` with
  `ad_lowering`.
- If any `fusion.*` value differs from `model_default`, merge `fusion` with
  `ad_lowering`.
- If a row uses DTensor placement for params, vectors, logits, tangents,
  cotangents, or outputs, merge `distributed_layout` with `ad_lowering`.
- If `fsdp.mp_policy.reduce_dtype` differs from `fp32`, merge
  `distributed_layout` with `numeric_backend`.
- If an inverse row uses a factorized metric or factorized preconditioner,
  merge `inverse_solve` with `metric_storage`.

Class C groups after merge must be searched jointly or with a staged method that
keeps top rows from each group before crossing groups.

### Search Strategies

`admission`:

- run static admission and backend availability checks
- build the candidate table
- run no full-size timings

`smoke`:

- measure one baseline row per operator family
- measure one representative row from each merged Class C group
- use tiny reference inputs and one full-size probe batch
- return a selected row only when all required references pass

`fast`:

- choose a baseline for each operator family
- run a coarse joint search over operator path, vectorization, dtype, attention,
  and compile-disabled eager execution
- use `autobatch` for microbatch, token block, vectorization batch, and chunk sizes
- compile only the top eager rows for each operator family
- keep one selected row per family plus the rejected rows needed to explain the
  choice

`balanced`:

- search each merged Class C group with successive halving
- keep the configured top count per group by reference-passing time and memory
- evaluate the bounded cross product of retained group winners
- compile top eager rows at multiple callable boundaries
- run selected-plan validation on full-size inputs

`thorough`:

- run balanced search
- expand top coupled groups with additional dtype, compile, layout, and
  chunking variants
- evaluate distributed rows across all declared ranks
- evaluate compile amortization across the declared call horizons
- repeat selected rows enough times to estimate variance

`exhaustive`:

- run the full admitted cross product
- allowed only for small models or explicitly bounded candidate sets

### Search State And Reuse

Search records enough state to resume without changing the search meaning:

- candidate settings
- operator spec
- axis owner identities
- integer domains and resolved integer values
- search strategy
- coupling-class decomposition
- successive-halving bracket state
- retained top counts
- declared call horizons for compile amortization
- measured recompile counts for compiled rows
- RNG seeds and stream state for stochastic rows
- backend availability
- target hardware identity
- input identity
- reference rows
- timing rows
- memory rows
- selected rows
- failed rows

Saved rows are reused only by direct field equality on the fields above.

## Measurement And Selection

Every row records:

- candidate settings
- executable lowering identity
- target hardware identity
- package version
- PyTorch version
- Transformers version when used
- backend availability checks
- warmup count
- measured call count
- elapsed samples
- peak allocated memory
- peak reserved memory
- post-call allocated memory
- post-call reserved memory
- output signature
- reference measurements
- failure type and message when failed

CUDA rows must either synchronize the device before and after the measured
region or use CUDA events that bracket the measured kernels. Distributed rows
must use rank barriers around the measured region and record global reductions
after all ranks complete.

Rows pass selection only when they:

- run without OOM
- run without graph-break failure when `compile.fullgraph=true`
- run without unsupported kernel fallback unless fallback is the declared row
- pass all reference checks
- pass full-size or kernel-triggering agreement checks for rows using non-math
  attention backends, shape-dependent attention backends, CUDA graphs,
  max-autotune compilation, fused kernels, packed kernels, sharded reductions,
  tensor-parallel loss, or context-parallel attention
- pass memory stability checks
- match direct replay fields for candidate, dependencies, runtime identity,
  target identity, and input identity

Within accepted rows, selection ranks by the configured objective:

- single-rank eager rows: median local steady-state elapsed time
- distributed eager rows: median rank-maximum steady-state elapsed time after barriers
- single-rank compiled rows: $((1+R)T_{\mathrm{compile}} / N) + T_{\mathrm{steady}}$
- distributed compiled rows: $((1+R)T_{\mathrm{compile}} / N) + T_{\mathrm{steady}}$ using rank-maximum compile time and rank-maximum steady-state time
- distributed memory tie breaking uses the selected reduction from `comm.rank_memory_reduction`
- ties inside the near-fastest band: lower peak reserved memory
- cohort comparison sums selected row scores and selected row memory scores

## Source Notes

This feature list is based on these current framework surfaces:

- PyTorch `torch.func`: `grad`, `grad_and_value`, `vjp`, `jvp`, `linearize`, `jacrev`, `jacfwd`, `hessian`, `vmap`, and `functional_call`.
- PyTorch transform limitations: pure-function requirements, `vmap` restrictions, mutation limits, randomness handling, and the restriction on mixing `torch.autograd.grad` inside transformed functions.
- PyTorch forward AD: `dual_level`, `make_dual`, and `unpack_dual`.
- PyTorch autograd functional APIs: `hvp` and `vhp`.
- PyTorch checkpointing: `use_reentrant`, `context_fn`, `determinism_check`, `debug`, `early_stop`, and `preserve_rng_state`.
- PyTorch SDPA: `scaled_dot_product_attention`, `sdpa_kernel`, and `SDPBackend`.
- PyTorch compile: `torch.compile`, compiled autograd, `fullgraph`, `dynamic`, `backend`, `mode`, `options`, and `torch.compiler.list_backends()`.
- PyTorch distributed: process groups, `DeviceMesh`, DTensor placements, DTensor redistribute, FSDP2 `fully_shard`, tensor parallel `parallelize_module`, and context parallel.
- Transformers: `attn_implementation`, `AttentionInterface`, `AttentionMaskInterface`, SDPA, FlashAttention-2, FlashAttention-3, FlexAttention, paged attention, and per-backbone attention maps.
