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
- loss and output-space Hessian for GGNVP
- score or log-prob definition for FisherVP
- per-example loss definition for empirical FisherVP
- metric definition and damping
- target devices and process count

The typed `loss`, `output`, `likelihood`, and `metric` objects carry these fixed
fields and validate their closed-set choices at construction.

Rows may change representations of fixed fields only when the row computes the
same operator and passes package-owned reference checks. Sequence packing,
padding changes, attention packing, teacher-output caching, and distributed
placement all fall under this rule.

Reference checks, numeric thresholds, numeric error bounds, and
memory-stability checks are acceptance rules. They are not speed knobs.

## Axis Ownership

Every setting key has exactly one owner, with one exception: `attention.frontend` is a
single key whose values split between the core attention executor (`pytorch_sdpa_direct`,
`patched_eager`, `packed_exact`, `blockwise_exact`) and the Transformers adapter (the
`transformers_*`, `paged|*`, and `registered_transformers_attention` values), so for it
the one-owner rule is read per value and the `AxisDescriptor` carries a per-value owner.
Duplicate owners on any other key are invalid.

- Operator-owned axes define the mathematical lowering for one operator family.
- Shared axes define implementation choices used by many operators.
- Adapter-owned axes are registered and lowered by an adapter. The model-library
  attention frontends (`transformers_*`, `paged|*`, `registered_transformers_attention`)
  and the whole distributed family (`distributed.*`, `dtensor.*`, `fsdp.*`, `tp.*`,
  `sequence_parallel.*`, `context_parallel.*`, `comm.*`) are adapter-owned; the core
  runtime does not lower them.
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
- per-example gradient
- metric multiply
- metric inner product
- metric square-root multiply
- inverse metric multiply
- inverse metric inner product
- inverse metric square-root multiply
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

For model output $z(\theta)$, a typed loss $\ell(z)$, Jacobian $J$, and vector $v$,
compute $Gv = J^\top H_\ell Jv$.

The typed loss and the output surface define the operator. Generalized Gauss-Newton is
defined for a loss convex in the output, so the output-space loss Hessian $H_\ell$ is
PSD and $G$ is PSD. Cross entropy, KL, retain KL, token masking, and reductions are
carried by the typed loss object as fixed problem fields.

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
- symmetry and PSD checks on the output-space loss Hessian $H_\ell$; a non-PSD $H_\ell$ fails admission
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

### Per-Example Gradient

For per-example loss, compute the stacked gradients $g_i=\nabla_\theta\ell_i(\theta)$. The
output is a parameter tree with a leading axis of size $n$, the batch example count; it is
not reduced.

Owned axes:

- `per_example_gradient.grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`
- `per_example_gradient.accumulation`: `stacked_leading_axis`, `blockwise_stacked`

The returned tree is the same fixed object for both `accumulation` values; only the
computation differs. `stacked_leading_axis` materializes all $n$ per-example gradients and
stacks them into the leading-axis tree. `blockwise_stacked` computes them in example-blocks
of `batch.per_example_block_size` and writes each block into the preallocated output,
bounding peak memory; it requires `batch.per_example_block_size`, which `stacked_leading_axis`
does not read. The per-example mapping is owned by `grad_path`; this family has no vector
dimension, so the `vectorization.*` axis does not apply.

Shared axes that apply: model call, batching and chunking (the example-block size is
`batch.per_example_block_size`), memory schedule, dtype and numeric backend, torch compile,
parameter and vector layout, distributed execution.

Mandatory checks:

- per-example gradients by for-loop reference
- agreement that the outer-product reduction equals empirical FisherVP
- full-size agreement for non-math or shape-dependent attention, compile, fusion, or
  sharded-reduction rows

### Metric Square-Root Multiply

For a metric $M$, apply a factor $Lv$ and its adjoint $L^\top v$, with $LL^\top=M$ for the
square root and $LL^\top=(M+\lambda I)^{-1}$ for the inverse square root. The
inverse-square-root operator applies $Lv$ for the damped-inverse factor, not $L^{-1}v$. The
factor is a covariance factor for posterior sampling, not the symmetric square root unless
an eigenbasis path is selected; the adjoint $L^\top v$ is the application the metric inner
product's `sqrt_apply_reduce` path consumes.

Owned axes:

- `sqrt_metric.factor_path`: `closed_form_factor_square_root`, `cholesky_factor`, `eigenbasis_factor`, `matrix_free_lanczos`
- `sqrt_metric.lanczos_iterations`: a finite positive integer domain (matrix-free path only)

Representation compatibility:

- `closed_form_factor_square_root` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors. It forms the forward factor directly ($A^{1/2}\otimes G^{1/2}$ for KFAC, the rooted corrected eigenvalues for EKFAC, $[U, D^{1/2}]$ for low-rank, $J^\top H_\ell^{1/2}$ for GGN-derived, the pointwise root for diagonal). For the damped inverse it serves diagonal, EKFAC (the corrected eigenvalues raised by `eigenvalue_floor`), KFAC under `kfac_pi` damping (whose factored shift keeps the Kronecker form $(A+\cdot)^{-1/2}\otimes(G+\cdot)^{-1/2}$), and low-rank and GGN-derived through the Woodbury capacitance.
- `cholesky_factor` requires a positive-definite dense or block-diagonal metric, and supplies the forward factor and a Cholesky of the dense damped inverse.
- `eigenbasis_factor` requires a symmetric dense, KFAC, or EKFAC metric. It is the path for the damped inverse square root over a KFAC metric with scalar or per-group damping, where the joint Kronecker spectrum is shifted by $\lambda$ before the inverse root and the un-shifted factor $A^{1/2}\otimes G^{1/2}$ cannot absorb the shift.
- `matrix_free_lanczos` requires a matrix-free metric and is the only path for it

Every KFAC and EKFAC metric is admitted for the forward and the damped inverse square root; the
rules above route each (metric kind, damping) pair to the factor path that produces the correct
factor, and the $LL^\top$ reference check rejects any other.

Shared axes that apply: batching and chunking, memory schedule, dtype and numeric
backend, torch compile, parameter and vector layout, distributed execution.

Mandatory checks:

- dense tiny factor check: $L L^\top$ matches $M$ or $(M+\lambda I)^{-1}$ on a reference
- a covariance check that $L z$ has the declared covariance on repeated draws for the
  matrix-free Lanczos path

### Metric Multiply

A metric operator is declared by its mathematical representation. Dense, KFAC,
EKFAC, diagonal, block diagonal, low rank, GGN-derived, and matrix-free metrics are
different metric specs unless the spec declares equivalence. The representation
supplies the required fields: dense matrix, diagonal tree, metric blocks, KFAC
factors, EKFAC Kronecker eigenbases with corrected eigenvalues, low-rank factors,
GGN-derived factors, or, for a matrix-free metric, the forward action of an admitted
PSD operator or composition (a GGN or Fisher). A matrix-free metric whose operator is
data-dependent takes the batch alongside the vector; the factored representations are
data-independent. The square-root and inverse-square-root multiplies apply a Cholesky
or eigenbasis factor for the factored kinds and a Lanczos approximation of $f(M)v$ for
the matrix-free kind.

Owned axes for a fixed metric spec:

- `metric.multiply_path`: `dense_matmul`, `factorized_multiply`, `blockwise_multiply`, `streaming_multiply`
- `metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`
- `metric.accumulation`: `streaming`, `materialized_blocks`

Representation compatibility:

- `dense_matmul` requires dense matrix representation
- `factorized_multiply` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors; EKFAC multiplies in the Kronecker eigenbasis
- `blockwise_multiply` requires block-diagonal blocks
- `streaming_multiply` requires diagonal, block-diagonal, KFAC, EKFAC, low-rank, or GGN-derived representation fields
- a matrix-free metric multiplies through its declared forward operator and uses none of `metric.multiply_path`, `metric.block_schedule`, or `metric.accumulation`
- `metric.block_schedule` requires block-diagonal blocks or KFAC factors
- `metric.accumulation` applies only to non-dense metric multiply paths

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny metric multiply
- dense reference reconstruction from the declared representation: dense matrix uses
  $M$ directly, diagonal tree assembles $\operatorname{diag}(d)$, block-diagonal
  blocks assemble $M=\operatorname{blockdiag}(M_1,\ldots,M_b)$, KFAC assembles
  each block as $A_b \otimes G_b$, low-rank assembles $M=UU^\top + D$, and
  GGN-derived assembles $M=J^\top H J$
- symmetry check when the metric is declared symmetric
- PSD check when the metric is declared PSD
- a matrix-free metric is checked through its operator's own anchors for the forward multiply and through the inverse residual for the solve, not through dense reconstruction

### Inverse Metric Multiply

For a declared or matrix-free metric $M$, compute $M^{-1}v$ or the declared damped
inverse. Damping and the accepted residual define the inverse metric spec.

Owned axes:

- `inverse_metric.solve_path`: `dense_solve`, `cholesky_solve`, `eigh_solve`, `svd_solve`, `conjugate_gradient`, `factorized_solve`, `blockwise_solve`, `woodbury_low_rank_solve`
- `inverse_metric.preconditioner`: `none`, `diagonal`, `block_diagonal`, `factorized_metric`, `matrix_free`
- `inverse_metric.iteration_budget`: declared positive integer set
- `inverse_metric.factor_reuse`: `refactor_each_rhs`, `reuse_factor_across_rhs`
- `inverse_metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`
- `inverse_metric.multi_rhs`: `single_column`, `block`

`inverse_metric.iteration_budget` is an implementation cap. A row with too low a
cap fails the fixed inverse residual acceptance check. Residual tolerance is the
operator's declared `tol`, a fixed acceptance field, not a sweep axis.
`inverse_metric.iteration_budget` applies only to iterative solve rows.

Representation compatibility:

- `dense_solve`, `cholesky_solve`, `eigh_solve`, and `svd_solve` require dense matrix representation
- `cholesky_solve`, `eigh_solve`, and `svd_solve` over a PSD-declared metric require a positive-definite operator (damping greater than zero); a PSD-but-singular GGN or Fisher with `damping=0` is rejected, because a zero eigenvalue or singular value makes the inverse undefined. `eigh_solve` additionally requires a symmetric metric
- `conjugate_gradient` requires an admitted metric multiply path for the same representation and a positive-definite metric, so a PSD-but-singular metric needs damping greater than zero; it is the only solve path for a matrix-free metric, inverting its forward operator iteratively
- `factorized_solve` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors; EKFAC inverts in the Kronecker eigenbasis by dividing the corrected eigenvalues
- `blockwise_solve` requires block-diagonal blocks
- `woodbury_low_rank_solve` requires low-rank factors
- `inverse_metric.preconditioner=block_diagonal` requires block-diagonal blocks or KFAC factors
- `inverse_metric.preconditioner=factorized_metric` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors
- `inverse_metric.preconditioner=matrix_free` wraps an admitted PSD operator, a tuned inverse-metric or factored-metric product named as the preconditioner
- `inverse_metric.block_schedule` requires block-diagonal blocks or KFAC factors
- `inverse_metric.multi_rhs=block` applies the solve to a stacked right-hand side, enabling block conjugate gradient and block Lanczos; `metric_vp` and `inverse_metric_vp` join the vectorization applicability for stacked vectors
- `dtype.metric_factor` and `memory.factor_residency` are declared only by rows whose metric or inverse path uses declared or computed factors

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

### Metric Inner Product

For a declared or matrix-free metric $M$ and two stacked vectors $U,V$ each $n\times k$, compute the $k\times k$ Gram $U^\top M V$. The $k=1$ case is the scalar $u^\top M v$. A generalized eigensolver reads the diagonal $v^\top M v$ as the squared $M$-norm for $M$-orthonormalization, and the operator's `as_norm` declaration requires an exactly nonnegative diagonal and pins the reduction path accordingly.

Owned axes:

- `metric_inner.reduction_path`: `multiply_then_reduce`, `factored_gram`, `sqrt_apply_reduce`
- `metric_inner.multi_rhs`: `single_column`, `block`

`multiply_then_reduce` applies the metric multiply to $V$ and forms $U^\top(MV)$; `factored_gram` forms the Gram from the metric factors in a Kronecker-aware order without materializing $MV$; `sqrt_apply_reduce` applies the square-root factor adjoint $L^\top$ (with $LL^\top=M$) to $U$ and $V$ and forms $(L^\top U)^\top(L^\top V)$, whose diagonal is $\|L^\top v\|^2\ge 0$ by construction. `as_norm=True` admits only `sqrt_apply_reduce`, because the tuner sweeps generic probes that never reach the near-null-space vectors where the other paths round to a negative diagonal.

Representation compatibility:

- `multiply_then_reduce` requires an admitted `metric.multiply_path` for the same representation
- `factored_gram` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors
- `sqrt_apply_reduce` requires an admitted `sqrt_metric.factor_path` for the metric and composes with it; a matrix-free metric uses the matrix-free Lanczos square root, whose Gram diagonal stays nonnegative under approximation
- `metric_inner.multi_rhs=block` batches the $k$ columns into one fused reduction and joins `metric_inner_vp` to the vectorization applicability

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny Gram against the reconstructed dense $M$: $U^\top M V$
- diagonal nonnegativity on the same-vector probe entries when `as_norm` is declared; the off-diagonal entries compare against the dense reference only
- symmetry of the Gram when $U=V$

### Inverse Metric Inner Product

For a declared or matrix-free metric $M$, two stacked vectors $U,V$ each $n\times k$, and the declared damping, compute the $k\times k$ Gram $U^\top (M+\lambda I)^{-1} V$. A generalized eigensolver reads the diagonal $r^\top (M+\lambda I)^{-1} r$ as the squared $R^{-1}$-norm of its residual, and the `as_norm` declaration requires an exactly nonnegative diagonal and pins the reduction path accordingly.

Owned axes:

- `inverse_metric_inner.reduction_path`: `solve_then_reduce`, `factored_gram`, `sqrt_apply_reduce`
- `inverse_metric_inner.multi_rhs`: `single_column`, `block`

`solve_then_reduce` solves $(M+\lambda I)X=V$ and forms $U^\top X$; `factored_gram` forms the Gram from the inverse factors, EKFAC through the corrected eigenvalues in the Kronecker eigenbasis; `sqrt_apply_reduce` applies the inverse-square-root factor adjoint $L^\top$ (with $LL^\top=(M+\lambda I)^{-1}$) to $U$ and $V$ and forms $(L^\top U)^\top(L^\top V)$, a forward factor application and not a solve, whose diagonal is $\|L^\top r\|^2\ge 0$. `as_norm=True` admits only `sqrt_apply_reduce`. The inverse inner product carries the operator's declared damping and residual tolerance `tol`.

Representation compatibility:

- `solve_then_reduce` requires an admitted `inverse_metric.solve_path` for the same representation and the same positive-damping requirement on a PSD-but-singular metric
- `factored_gram` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors
- `sqrt_apply_reduce` requires an admitted `sqrt_metric.factor_path`; a matrix-free metric uses the matrix-free Lanczos inverse square root, whose Gram diagonal stays nonnegative under approximation
- `inverse_metric_inner.multi_rhs=block` batches the $k$ columns and joins `inverse_metric_inner_vp` to the vectorization applicability

Shared axes that apply:

- batching, chunking, and input representation
- memory schedule
- dtype and numeric backend
- torch compile
- parameter and vector layout
- distributed execution

Mandatory checks:

- dense tiny Gram against the reconstructed damped inverse: $U^\top (M+\lambda I)^{-1} V$
- diagonal nonnegativity on the same-vector probe entries when `as_norm` is declared
- inverse residual on each solved column for `solve_then_reduce`
- positive damping over a PSD-but-singular metric

### Composition

For child operators arranged by an operator expression, compute the declared
composition. The expression has two node types and two leaves: `compose` gives the
sequential application $e_1(e_2(\cdots e_k(v)))$, `linear_combination` gives the
weighted sum $\sum_i c_i e_i(v)$, the `scaled_identity` leaf gives $c v$, and the
`source` leaf seeds $v$ from a batch-to-vector child. `compose` and `linear_combination`
close the linear operator algebra under composition, addition, and scalar
multiplication; `scaled_identity` supplies the identity, and `source` seeds the vector so a
source-bearing expression is vector-valued, a generator of $v$ rather than an operator on it.
Preconditioning is a `compose`, damping and averaging are a `linear_combination`, the
natural-gradient step is a `compose` over a `source`, and fused composites are the
`fuse_adjacent_children` lowering of a `compose` node. A `source` may appear only as the
innermost argument of a `compose` or a term of a `linear_combination`; an expression with
a source is vector-valued. Inversion is the inverse-metric operator's responsibility: a
PSD matrix-free composite, a GGN or Fisher, is inverted by wrapping it as a matrix-free
metric and solving with conjugate gradient, which requires the PSD operator.

The child-name leaves of the expression are the single source for the composition
family's dependencies. Composition requires a multi-family run because child
families must be present in the same run-level dependency graph.

A `compose` node lowers through the `sequential_composition` runtime path; a
`linear_combination` node lowers through the `linear_combination` runtime path,
applying its terms to the same input vector and reducing them with the declared
coefficients. `composition.execution` arranges a `compose` node's children.

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
FisherVP, metric multiply, inverse metric multiply, the metric inner products, the
square-root multiplies, and any composition child that accepts multiple vectors or
cotangents. For metric multiply, inverse metric multiply, and the metric inner products it
carries the stacked right-hand side that `inverse_metric.multi_rhs=block`,
`metric_inner.multi_rhs=block`, and `inverse_metric_inner.multi_rhs=block` consume. This
axis owns the vector, tangent, or cotangent dimension only. It does not
own data-example batching, per-example gradient batching, token packing, or
microbatching.

- `vectorization.mode`: `single_loop`, `manual_batch`, `vmap`
- `vectorization.batch_size`
- `vectorization.vmap_chunk_size`
- `vectorization.in_dims`
- `vectorization.randomness`: `error`, `same`, `different`

`vmap` rows require transform-compatible code and explicit randomness behavior.

### Derived Gradient Materialization

Applies to gradient, VJP, and HVP rows that compute gradients with respect to the
declared parameter surface.

Gradient materialization is derived from the AD path. It is recorded on rows, but
it is not a sweep axis.

- `gradient.path=backward_materialized_grad` and
  `vjp.path=backward_materialized_grad`: materialize `.grad` and read it back.
- All `torch.func` rows and eager `torch.autograd.grad` rows: return a tensor
  tree.

Clearing stale `.grad` values before probes is mandatory execution hygiene and
is never a row.

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

Core attention execution owns `pytorch_sdpa_direct`, `patched_eager`,
`packed_exact`, `blockwise_exact`, `attention.sdpa_kernel`,
`attention.partition`, and `attention.padding`. A model adapter supplies an
attention-location descriptor. The Transformers adapter owns
`transformers_*`, `paged|*`, and `registered_transformers_attention` frontends and
enters them into the search space through `space.with_attention(...)`.

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
- `batch.per_example_block_size`
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
`attention.partition=segmented_forward_ad` requires a forward-AD operator path.

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
- `numeric.float32_matmul_precision=high`
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
- `compile.boundary`: `model_forward`, `transformer_block`, `attention_module`, `loss_closure`, `gradient_closure`, `jvp_closure`, `vjp_closure`, `hvp_single_vector`, `hvp_batched_vectors`, `ggn_jvp`, `ggn_loss_hessian_product`, `ggn_vjp`, `ggn_full_product`, `fisher_score_grad`, `sampled_fisher_score_grad`, `empirical_fisher_example_grad`, `metric_multiply`, `metric_inner_reduce`, `metric_sqrt_multiply`, `inverse_metric_solve`, `inverse_metric_inner_reduce`, `per_example_gradient`, `bound_operator_vector_step`, `composition_child`, `whole_operator`
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

Rows that set any `compile.options.*` value to `true` must use
`compile.mode=None`. Rows with all compile options disabled must not set
`compile.mode=None`.

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
DTensor placement and distributed output placement are owned here. The whole
distributed family is owned by the distributed adapter (`vptune.adapters.distributed`)
and enters the search space through `space.with_distributed(...)`; the core runtime
does not lower it.

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
  `fisher.*`, `sampled_fisher.*`, `empirical_fisher.*`, `per_example_gradient.*`,
  `composition.*`, `vectorization.*`, and `call.*`
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
- `metric_storage`: `metric.*`, `metric_inner.*`, `sqrt_metric.*`, and
  `memory.factor_residency`
- `inverse_solve`: `inverse_metric.*` and `inverse_metric_inner.*`

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
- If an `inverse_metric_inner` row uses `factored_gram` or `sqrt_apply_reduce`,
  merge `inverse_solve` with `metric_storage`, since the reduction reads the same
  factors.

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
- distributed memory tie breaking uses the selection policy's rank-memory reduction
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
