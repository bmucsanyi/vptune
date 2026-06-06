# vptune Specification

## Purpose

`vptune` is a PyTorch package that takes a model, data, vectors, an operator spec, and a hardware target, then selects the fastest numerically stable implementation that fits memory.

`vptune` supports the pilot through an adapter. Core requirements come from derivative-operator tuning: declared semantics, package-owned anchors, candidate search, measurement, stability checks, and selected-plan replay.

`vptune` owns:

- Operator families: gradient, JVP, VJP, HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, per-example gradient, metrics, metric inner products, metric square roots, inverse metrics, inverse metric inner products, inverse metric square roots, and compositions.
- Package-owned anchors for standard operator families.
- Candidate generation, candidate validation, candidate dependency ordering, and candidate admission checks.
- Execution strategies: forward-over-reverse, reverse-over-reverse, reverse-over-forward, `vmap` batched transforms, row loops, microbatching, recomputation, checkpointing, model dtype, compute dtype, the model-agnostic attention executor, and tensor layout. The model-library attention frontends and sharding are adapter-owned.
- Measurement: elapsed time, peak allocated memory, peak reserved memory, post-call allocated memory, post-call reserved memory, OOM status, runtime failure status, and repeated-call memory behavior.
- Selection: fastest stable row, then lower peak reserved memory among near-fastest rows.
- Saved records: inputs, candidates, checks, measurements, selected settings, and selected-plan validation results. Replay uses direct field equality only.

The package goal is direct: turn "compute this matrix-free derivative-vector product on this model, data, vectors, and accelerator" into the fastest stable implementation that runs.

## Design Principles

`vptune` is an autotuner for matrix-free derivative operators. The core package stays small:

- Define the mathematical operator.
- Define the parameter surface and data aggregation.
- Generate candidate settings.
- Run package-owned reference checks.
- Measure speed and memory.
- Select a stable setting.
- Save enough identity to replay the choice.

The pilot is the first demanding use case. It guides required capabilities, but it does not define package internals. Current pilot row ids, parent chains, reason strings, stage names, and summary layout are migration details handled at the edge.

The reusable core should expose general mechanisms:

- Family DAGs instead of pilot stages.
- Candidate settings instead of hand-coded row labels.
- Adapter admission rules instead of model-specific branches in core.
- Operator anchors instead of caller-only reference harnesses.
- Selection policies instead of hard-coded timing and memory tie rules.
- Stable JSON schemas instead of pilot summary coupling.

## Source Basis

This spec is grounded in the current repo, `FEATURES.md`, and current framework docs. PyTorch stable docs redirect to PyTorch 2.12 on 2026-06-01.

Local pilot sources were read to validate that a real adapter can express an LLM tuning workload. The core spec does not take package internals from pilot row names, parent chains, stage names, or summary layout.

Framework sources:

- [PyTorch `torch.func` API](https://docs.pytorch.org/docs/2.12/func.api.html): `grad`, `vjp`, `jvp`, `linearize`, `jacrev`, `jacfwd`, `hessian`, `vmap`, and `functional_call`.
- [PyTorch `functional_call`](https://docs.pytorch.org/docs/2.12/generated/torch.func.functional_call.html): functional module calls and tied-weight handling through `tie_weights`.
- [PyTorch `jvp`](https://docs.pytorch.org/docs/2.12/generated/torch.func.jvp.html): standard JVP transform and forward-mode coverage errors.
- [PyTorch `vjp`](https://docs.pytorch.org/docs/2.12/generated/torch.func.vjp.html): standard VJP transform.
- [PyTorch `hessian`](https://docs.pytorch.org/docs/2.12/generated/torch.func.hessian.html): forward-over-reverse default and reverse-over-reverse coverage option.
- [PyTorch `torch.func` UX limitations](https://docs.pytorch.org/docs/2.12/func.ux_limitations.html): transform purity, `vmap` restrictions, mutation, dynamic shape limits, and randomness handling.
- [PyTorch forward AD](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.forward_ad.enter_dual_level.html): dual levels, `make_dual`, and `unpack_dual`.
- [PyTorch `torch.autograd.functional.hvp`](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.functional.hvp.html): scalar HVP reference API.
- [PyTorch `torch.autograd.functional.vhp`](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.functional.vhp.html): scalar VHP reference API.
- [PyTorch checkpointing](https://docs.pytorch.org/docs/2.12/checkpoint.html): non-reentrant checkpointing supports `torch.autograd.grad` and keyword arguments.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html): allocated memory versus reserved allocator memory.
- [PyTorch reset peak memory stats](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_peak_memory_stats.html): peak memory reset API.
- [PyTorch device memory used](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.device_memory_used.html): global device memory reported by `nvidia-smi` or `amd-smi`.
- [PyTorch SDPA](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html): scaled dot-product attention and backend dispatch.
- [PyTorch SDPA backend context](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html): `sdpa_kernel(backends, set_priority=False)` and priority-ordered backend lists.
- [PyTorch `SDPBackend`](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.SDPBackend.html): `MATH`, `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, and `OVERRIDEABLE`.
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html): `backend`, `mode`, `fullgraph`, `dynamic`, backend options, CUDA graph modes, and max-autotune modes.
- [PyTorch compiled autograd](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html): capturing larger backward graphs under `torch.compile`.
- [PyTorch DTensor](https://docs.pytorch.org/docs/2.12/distributed.tensor.html): `DeviceMesh` and placements.
- [PyTorch FSDP2](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html): `fully_shard`, all-gather, reduce-scatter, and prefetch behavior.
- [PyTorch tensor parallelism](https://docs.pytorch.org/docs/2.12/distributed.tensor.parallel.html): colwise, rowwise, and sequence-parallel styles.
- [Transformers attention backends](https://huggingface.co/docs/transformers/attention_interface): `attn_implementation`, `AttentionInterface`, `AttentionMaskInterface`, FlashAttention-2, FlashAttention-3, FlexAttention, paged attention, registered kernels, and `set_attn_implementation`.
- [Transformers continuous batching](https://huggingface.co/docs/transformers/main/continuous_batching): paged attention backend strings.

`FEATURES.md` defines the full sweep space. `SPEC.md` defines the package objects, schema fields, executor interfaces, reference checks, search flow, replay, and tests that implement that sweep space. Every axis key in `FEATURES.md` must appear in the package axis manifest with one owner, one value domain, one search group, and one lowering or admission rule.

## First Adapter Target

The pilot adapter proves that `vptune` can handle a real LLM derivative-operator tuning workload. It drives missing general capabilities into the package core, but pilot row names and stage layout stay in `vptune.adapters.pilot`.

The adapter acceptance target requires:

- A family DAG over standard operators, custom objectives, and adapter-owned composite computations.
- Cohort constraints keyed by declared settings. Dtype coherence is one use case.
- Candidate axes registered by the adapter for its useful search region.
- Package-owned references for the standard operator families.
- Custom caller objectives for research quantities.
- Full-size measurement and stable-memory filtering.
- Selected-plan validation using selected dependency outputs.
- An adapter that converts a `vptune` plan into caller settings.

The adapter registers its search region as named candidate axes. Domain names such as chart verification chunks, retain minibatch chunks, and contact row counts stay in the adapter layer. Core sees ordinary candidate settings with explicit axis ownership, admission checks, and dependency selections.

The adapter acceptance gate checks behavior:

- Every declared computation has a selected package candidate.
- Selected settings are convertible to the caller settings reader.
- Selected-plan validation passes using the converted selected settings.
- Selected-settings conversion rejects plans that require validation unless matching selected-plan validation rows passed.
- Downstream consumers see selected dependency readiness through the adapter readiness API.
- Reference checks, numeric thresholds, input signatures, and replay identity fields match the converted settings by direct field equality.

## Package Boundary

Core package:

- Standard operator semantics.
- Built-in anchors.
- Standard runtime builders for package-owned operators.
- Candidate rows.
- Candidate validation.
- Measurement.
- Memory stability.
- Selection.
- Saved records.
- `autobatch` integration.

Adapter modules:

- `vptune.adapters.transformers`: Hugging Face model loading, the attention frontends and the `attention_space(...)` they register, tied-weight handling, cache flags, tokenizer-aware batching, and model-specific attention variants.
- `vptune.adapters.distributed`: the `target(...)`/`space(...)` builders for DTensor, FSDP2, tensor parallel, sequence and context parallel, rank-local memory, global status, and selected settings agreement.
- `vptune.adapters.pilot`: pilot lowering, readiness, selected settings conversion, and adapter-owned validation.

`import vptune as vp` exports the user-facing surface: the model builders `torch_model`, `parameters`, and `module_call`; the typed math objects `output`, `loss.*`, `likelihood.*`, `metric.*`, `samples.*`, and `damping.*`; the operator constructors, the returned `Operator`, and the composition combinators `compose`, `linear_combination`, `scaled_identity`, and `source`; the `space`, `search`, `cuda`, `cohort.*`, and `CohortConstraint` builders; `tune`; the lower-layer `problem`, `autotune`, `Plan`, `TuningRun`, `tune_run`, and replay functions; and the package errors. `Candidate`, `RuntimeConfig`, the axis table and manifest, operation factories, reference checks, package-owned anchors, memory backends, Autobatch domains, and the schema and replay helpers live in `vptune.ext`; a normal caller never touches them. Adapter helpers are imported from `vptune.adapters` or the specific adapter module.

Non-goals. The package tunes matrix-free matvecs and the factor, inverse, square-root, and inner-product products built from them; it does not own the algorithms that orchestrate them. The boundary is the tuning surface: a product is in scope when it has an implementation choice beyond the matvec it sits on (the inverse solve path, the square-root factor path, the inner-product reduction path and its block batching), and out of scope when it is loop orchestration with no such choice. Out of scope: eigensolvers and Krylov drivers (Lanczos, LOBPCG, CG loops) that sequence the tuned products; indefinite-Hessian inversion (CG requires a PD operator and the package ships no MINRES or SYMMLQ, so the indefinite object stays `hvp`); non-PyTorch backends; and any sampling scheme without a fixed-seed or fixed-table identity. The $R$-norm and $R^{-1}$-norm Gram matrices a generalized eigensolver consumes are tuned products, `metric_inner_vp` and `inverse_metric_inner_vp`; the eigensolver loop that calls them is the caller's.

The live package spec is `SPEC.md`. The live package scratchpad is `SCRATCHPAD.md`.

## Public API

The public surface declares mathematical objects. A caller declares a model, what to differentiate, the vectors and data, a search space, and a target. The operator is the front-door object: it is callable on a default implementation immediately, and tuning returns a faster callable that carries its selection record. `Candidate`, `RuntimeConfig`, the axis table, anchors, and schema helpers live in `vptune.ext`; a normal caller never writes a candidate row.

### Model

```python
import vptune as vp

model = vp.torch_model(
    module,
    parameters=vp.parameters(module, include=include_rule, buffers="include", tied="preserve"),
    call=vp.module_call(args=("input_ids",), kwargs={"attention_mask": "attention_mask"}, output="logits"),
)
```

`vp.parameters(...)` is the typed parameter surface: active set, trainable flags, tied-weight and buffer policy, and the canonical flatten order. `vp.module_call(...)` is the batch-to-`forward` binding. Together with the module they carry everything the front door previously asked for through `parameter_surface`, `parameter_values`, `buffers`, and objective wiring.

### Typed inputs that carry the math

A typed object carries the mathematical content each operator needs and validates its closed-set fields at construction:

- `vp.output(field)` is the vector output function $z(\theta)$ that JVP and VJP differentiate; `field` names a key in the model-call output, for example `vp.output("logits")`. The `output=` field on a loss or likelihood names the same model-output key that object reads.
- `vp.loss.*` is a scalar loss that also exposes its output-space Hessian $H_\ell$. `vp.loss.softmax_cross_entropy(output="logits", labels="labels", mask=None, reduction="token_mean", denominator="num_tokens")` has the per-token block $\operatorname{diag}(p_t)-p_t p_t^\top$, and the operator-level $H_\ell$ is the masked, normalized sum $\frac{1}{N}\sum_{m_t=1}(\operatorname{diag}(p_t)-p_t p_t^\top)$ with $N=\sum_t m_t$ and a zero block at masked positions. `vp.loss.kl(...)` and `vp.loss.mse(...)` follow. `vp.loss.from_scalar(fn, output=..., version=...)` takes $H_\ell$ from autodiff, PSD-checked at the probe. `vp.loss.declared_psd(output=..., factors=...)` is PSD by construction; `vp.loss.declared_psd_matrix_free(output=..., matvec=..., version=...)` is PSD-checked by a Lanczos eigenvalue estimate with a residual bound, admitted only when $\theta_1-\|r_1\|\ge-\tau$ for the smallest Ritz value $\theta_1$ and its residual norm $\|r_1\|$, or refused. `reduction` is `sum`, `mean`, `token_mean`, or a named adapter reduction; `denominator` aligns with the likelihood denominator so CE-GGN and exact Fisher coincide.
- `vp.likelihood.*` is the predictive distribution defining the score $s_\theta=\nabla_\theta\log p_\theta(y\mid x)$: `vp.likelihood.categorical(output="logits", labels="labels", sample_space="terms", denominator="num_tokens", label_policy="explicit")` and `vp.likelihood.gaussian(output=..., target=..., noise=..., sample_space=..., denominator=...)`. The label policy, sample space, score reduction, and denominator that were string arguments are validated closed-set fields here.
- `vp.metric.*` is a typed metric: `vp.metric.dense(matrix=...)`, `vp.metric.diagonal(diag=...)`, `vp.metric.block_diagonal(blocks=...)`, `vp.metric.kfac(factors=..., dampings=None)`, `vp.metric.ekfac(eigvecs_a=..., eigvecs_g=..., corrected_eigenvalues=...)`, `vp.metric.low_rank(factor=..., diagonal=...)`, `vp.metric.ggn_derived(factors=...)`, and `vp.metric.matrix_free(operator=...)` where `operator` is a sibling product, an `Operator` carrying a `name`. The metric records the product name so dependency and cohort machinery reach it, and an anonymous operator is rejected; its operator must be an admitted PSD curvature (a GGN or Fisher), inverted through conjugate gradient with positive damping. It replaces the opaque `representation: Mapping[str, Any]`.
- `vp.samples.*` declares a sampled-Fisher sample source: `vp.samples.fixed_seed(seed, count)` or `vp.samples.table(table=..., identity=...)`.
- `vp.damping.*` is the typed damping for `inverse_metric_vp`, `inverse_metric_inner_vp`, and `inverse_sqrt_metric_vp`: `vp.damping.scalar(lam)`, `vp.damping.per_group(values)` keyed by parameter-surface block, `vp.damping.kfac_pi(lam, policy="trace_norm" | "equal")` for the Martens-Grosse factor split, and `vp.damping.eigenvalue_floor(lam)` for EKFAC. `kfac_pi` requires a `vp.metric.kfac` metric (it needs the separate $A$ and $G$ factor norms, which EKFAC discards); `eigenvalue_floor` requires `vp.metric.ekfac` and adds $\lambda$ to the corrected eigenvalues; admission rejects the other pairings.

### The operator is the object

```python
ggn = vp.ggnvp(model, vp.loss.softmax_cross_entropy(output="logits", labels="labels", mask="attention_mask"))
y = ggn(batch, vector)             # callable now on a default reference implementation

ggn = ggn.tune(                    # returns a new tuned operator carrying its record
    data=probe_batches, vectors=probe_vectors,
    target=vp.cuda(0, "h100"),
    space=vp.space.standard(...).with_attention(...),
    search=vp.search.balanced(retain=4),
    run_dir=run_dir,
)
y = ggn(batch, vector)             # fastest stable implementation
ggn.plan                           # selection, measurements, replay record
ggn = vp.ggnvp(model, loss).load(run_dir)   # replay a saved tuning without re-searching
```

The call signature matches the operator. `gradient` takes a batch. `jvp`, `vjp`, `hvp`, `ggnvp`, and the three Fishers take a batch and a vector. `metric_vp` and `inverse_metric_vp` over a declared, factored metric take only a vector, because that metric is data-independent; over a `vp.metric.matrix_free(...)` metric wrapping a data-dependent operator they take a batch and a vector. `tune` returns a new operator and does not mutate in place; every default it applied is recorded in `ggn.plan`.

### Data and vectors

`data` is an iterable of representative batches, or a mapping by product name when products consume different data; `vectors` is an iterable for a single product or a mapping by product name for several products. A composition with per-child data (influence functions with a test and a train batch, forget and retain losses on disjoint subsets, function-space Laplace with distinct evaluation batches) routes a named data stream to each child by the child's product name. The tuner forms probe `(batch, vector)` pairs and derives a tiny reference pair for correctness checks. The overrides are `reference=vp.case(batch=..., vector=...)` for the correctness input and `probes=[(batch, vector), ...]` for exact control. This replaces the `DataProvider`/`VectorProvider` providers zipped by position. A data-independent operator such as `metric_vp` over a factored metric takes vectors with no batch. `cohort_constraints` pins sweep axes that several products must agree on; the pilot pins `layout.vector`, `layout.flatten_order`, and `dtype.vector` across its $H$, $R$, and $R^{-1}$ products so a generalized eigensolver can feed one vector to all three.

### Search space, target, and search

```python
space = vp.space.standard(
    autodiff=vp.AD(paths=("torch_func_jvp", "forward_ad_dual")),
    vectorization=vp.Vectorization(modes=("single_loop", "vmap", "manual_batch")),
    precision=vp.Precision(model=("fp32", "bf16"), accumulation=("fp32", "bf16")),
    compile=vp.Compile(enabled=(False, True), boundaries=("ggn_full_product",)),
    layout=vp.Layout(params=("flat_contiguous",), vector=("flat_contiguous",)),
    memory=vp.Memory(vector_residency=("gpu",), recompute=("none",)),
)
space = space.with_attention(vp.adapters.transformers.attention_space(frontends=("sdpa", "flash_attention_2")))
space = space.with_distributed(vp.adapters.distributed.space(strategy=("fsdp2",)))

target = vp.cuda(
    device=0, accelerator="h100",
    timing=vp.TimingPolicy(), selection=vp.SelectionPolicy(),
    determinism=vp.DeterminismPolicy(), environment=vp.EnvironmentPolicy(),
)
search = vp.search.balanced(retain=4, compile_horizons=(1, 10, 100))   # .fast/.thorough/.smoke/.admission/.exhaustive
```

`vp.space.standard(...)` declares only the axes the core runtime lowers. Attention frontends and distributed strategies are adapter-owned and enter the search space through `space.with_attention(...)` and `space.with_distributed(...)`. `vp.cuda(...)` is the hardware target with its timing, selection, determinism, and environment-capture policies. `vp.search.balanced(...)` and its `.fast`, `.thorough`, `.smoke`, `.admission`, and `.exhaustive` siblings select the search strategy.

### Multiple products

Several products tuned in one run (cohorts, compositions) use `vp.tune`, which returns tuned callables by name:

```python
run = vp.tune(
    products=[curvature, preconditioner, preconditioned_hvp],   # composition children are sibling products
    model=model, data=probe_batches,
    vectors={"curvature": [...], "preconditioner": [...], "preconditioned_hvp": [...]},
    target=vp.cuda(0, "h100"),
    search=vp.search.balanced(retain=4, compile_horizons=(1, 10, 100)),
    space=space, run_dir=run_dir,
)
precond = run["preconditioned_hvp"]         # tuned callable
```

### The lower layer

`operator.tune(...)` and `vp.tune(...)` build a `Problem` (single product) or a `TuningRun` (several products) and run the search under it. Advanced callers reach that layer directly: `vp.problem(product, ...)` builds a single-product `Problem`, `vp.autotune(problem, run_dir=...)` runs the search and returns a `Plan`, and `plan.materialize(name)` returns the selected callable. A `composition` is an `Operator`, but its children are sibling products in a run-level dependency graph, so a composition is tuned only through `vp.tune(...)` or a `TuningRun`; `operator.tune(...)` on a composition, and `vp.problem`/`vp.autotune` on a composition spec, are rejected. Custom operators and adapters can provide `vptune.ext.RuntimeConfig` directly with an operation factory, reference check, materializer, axis registry, and runtime identity.

`vp.tune(products=...)` returns a `Run`, a mapping from product name to tuned `Operator`; it builds and runs a `TuningRun` internally and exposes the `Run` view. The `TuningRun` plus `tune_run` path below is the lower layer a caller reaches directly only to supply adapter-lowered `problems`, `validators`, and `validator_identities`:

```python
import vptune as vp

loss = vp.loss.softmax_cross_entropy(output="logits", labels="labels")
curvature = vp.hvp(model, loss, name="curvature")
preconditioner = vp.inverse_metric_vp(model, vp.metric.kfac(factors=kfac), damping=vp.damping.scalar(1e-2), name="preconditioner")
preconditioned_hvp = vp.composition(
    model,
    children=("preconditioner", "curvature"),
    combine=vp.compose("preconditioner", "curvature"),
    name="preconditioned_hvp",
)

tuning = vp.TuningRun(
    target=target,
    products=(curvature, preconditioner, preconditioned_hvp),
    cohort_constraints=(vp.cohort.layout_coherence(("layout.vector", "layout.flatten_order", "dtype.vector")),),
    problems=adapter.lower(tuning_inputs),
    validators=adapter.validators(tuning_inputs),
    validator_identities=adapter.validator_identities(tuning_inputs),
    run_id="example-run",
)

plan = vp.tune_run(tuning, run_dir=run_dir)
```

When `TuningRun.validators` is non-empty, `TuningRun.validator_identities` must cover the same products with non-empty identities. `tune_run` rejects validator key mismatches before search, writes the selected plan with `validation_required=True`, runs selected-plan validation in `Plan.validation_order`, writes validation records and `summaries/selected_plan_validation.json`, and raises if any selected product fails validation.

Replay has a front door and a lower layer, and each entry point reads one kind of saved run. `vp.<op>(model, spec).load(run_dir)` replays a saved single-product tuning into a callable; it is the only door for a run produced by `operator.tune(...)`. `vp.load_tuned_plan(run_dir, problem)` replays a saved single-product run produced by `vp.problem`/`vp.autotune`, and `vp.load_tuned_run(run_dir, tuning)` replays a saved `TuningRun` produced by `vp.tune`/`tune_run`, both without a caller-built replay identity. `vp.load_plan(run_dir, replay_context, materializers)` is the manual door for callers that supply their own replay context. The package reads the saved run kind and raises when an entry point is handed a run it does not cover; the kinds do not overlap.

Every public object is typed and serializable. Every replay-relevant identity is saved as explicit fields, and a selected run can be reproduced from saved records without re-running search.

Concrete public signatures:

```python
def torch_model(
    module: torch.nn.Module,
    *,
    parameters: ParameterSurface,
    call: ModuleCallSpec,
) -> Model: ...

def parameters(
    module: torch.nn.Module,
    *,
    include: Callable[[str, torch.nn.Parameter], bool] | None = None,
    buffers: str = "include",
    tied: str = "preserve",
) -> ParameterSurface: ...

def module_call(
    *,
    args: Sequence[str],
    kwargs: Mapping[str, str],
    output: str,
) -> ModuleCallSpec: ...
```

Every operator constructor returns an `Operator`. The operator binds its model, is callable on a default implementation, and tunes into a faster callable that carries its `Plan`:

```python
class Operator(Protocol):
    plan: Plan | None
    call_inputs: tuple[str, ...]   # ("batch",) | ("batch", "vector") | ("vector",) | ("left", "right") | ("batch", "left", "right")

    # gradient(batch); jvp/vjp/hvp/ggnvp/fisher*(batch, vector);
    # metric_vp/inverse_metric_vp(vector) for a factored metric, (batch, vector) for matrix_free
    # metric_inner_vp/inverse_metric_inner_vp(left, right) for a factored metric, (batch, left, right) for matrix_free
    def __call__(self, *call_inputs: Any) -> Any: ...

    # fix the batch for repeated calls (a Krylov solver wants op(vector)); the result has call_inputs ("vector",)
    def bind(self, *, batch: Batch) -> Operator: ...

    def tune(
        self,
        *,
        data: Iterable[Batch] | None = None,
        vectors: Iterable[TensorTree] | None = None,
        target: Target,
        space: SearchSpace,
        search: SearchStrategy,
        run_dir: Path | None = None,
        reference: Case | None = None,
        probes: Sequence[tuple[Batch, TensorTree]] | None = None,
        memory_backend: vptune.ext.MemoryBackend | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> Operator: ...

    def load(
        self,
        run_dir: Path,
        *,
        memory_backend: vptune.ext.MemoryBackend | None = None,
    ) -> Operator: ...
```

```python
def tune(
    *,
    products: Sequence[Operator],
    model: Model,
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
    vectors: Mapping[str, Iterable[TensorTree]],
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
    cohort_constraints: Sequence[CohortConstraint] = (),
    run_dir: Path | None = None,
    memory_backend: vptune.ext.MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Run: ...

def problem(
    product: Operator,
    *,
    data: Iterable[Batch],
    vectors: Iterable[TensorTree],
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
) -> Problem: ...

def autotune(
    problem: Problem,
    *,
    run_dir: Path | None = None,
    memory_backend: vptune.ext.MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Plan: ...

def tune_run(
    run: TuningRun,
    *,
    run_dir: Path,
    memory_backend: vptune.ext.MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Plan: ...

def materialize(plan: Plan, *, name: str | None = None) -> Any: ...

def load_tuned_plan(
    run_dir: Path,
    problem: Problem,
    *,
    memory_backend: vptune.ext.MemoryBackend | None = None,
) -> Plan: ...

def load_tuned_run(
    run_dir: Path,
    run: TuningRun,
    *,
    memory_backend: vptune.ext.MemoryBackend | None = None,
) -> Plan: ...

def load_plan(
    run_dir: Path,
    *,
    replay_context: ReplayContext,
    materializers: Mapping[str, Materializer],
) -> Plan: ...

def validate_plan(
    plan: Plan,
    validators: Mapping[str, PlanValidator],
    *,
    run_dir: Path | None = None,
) -> tuple[CheckRecord, ...]: ...
```

Operator constructor signatures:

Every operator except `composition` follows one shape, `(model, spec, name=None, **op_specific)`, and returns an `Operator`. The spec is the typed object that carries the math: a `Loss` for `gradient`, `hvp`, `ggnvp`, `empirical_fisher_vp`, and `per_example_gradient`; an `Output` for `jvp` and `vjp`; a `Likelihood` for `fisher_vp` and `sampled_fisher_vp`; a `Metric` for `metric_vp`, `metric_inner_vp`, `inverse_metric_vp`, `inverse_metric_inner_vp`, `sqrt_metric_vp`, and `inverse_sqrt_metric_vp`. `composition` is the explicit exception: it carries no single spec but a `children` list and a `combine` expression. `name` is optional for a single product and required when several products tune together. Each operator declares its `call_inputs`, the ordered call arguments, one of `("batch",)`, `("batch", "vector")`, `("vector",)`, `("left", "right")`, or `("batch", "left", "right")`, so the call arity is inspectable and not a hidden mode.

```python
def gradient(model: Model, loss: Loss, name: str | None = None) -> Operator: ...
def jvp(model: Model, output: Output, name: str | None = None) -> Operator: ...
def vjp(model: Model, output: Output, name: str | None = None) -> Operator: ...
def hvp(model: Model, loss: Loss, name: str | None = None) -> Operator: ...
def ggnvp(model: Model, loss: Loss, name: str | None = None) -> Operator: ...
def fisher_vp(model: Model, likelihood: Likelihood, name: str | None = None) -> Operator: ...

def sampled_fisher_vp(
    model: Model,
    likelihood: Likelihood,
    name: str | None = None,
    *,
    samples: SampleSource,
) -> Operator: ...

def empirical_fisher_vp(model: Model, loss: Loss, name: str | None = None) -> Operator: ...

def per_example_gradient(model: Model, loss: Loss, name: str | None = None) -> Operator: ...

def metric_vp(model: Model, metric: Metric, name: str | None = None) -> Operator: ...

def metric_inner_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    as_norm: bool = False,
) -> Operator: ...

def sqrt_metric_vp(model: Model, metric: Metric, name: str | None = None) -> Operator: ...

def inverse_metric_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    tol: float | None = None,
) -> Operator: ...

def inverse_metric_inner_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    as_norm: bool = False,
    tol: float | None = None,
) -> Operator: ...

def inverse_sqrt_metric_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    tol: float | None = None,
) -> Operator: ...

def composition(
    model: Model,
    name: str | None = None,
    *,
    children: Sequence[str],
    combine: Combine,
) -> Operator: ...
```

`ggnvp` takes a loss (or `vp.loss.declared_psd(...)`) and enforces PSD on the output-space Hessian $H_\ell$: a non-PSD $H_\ell$ fails admission with "GGN requires a PSD output-space metric." There is no `loss_geometry` and no `linear_map`. Exact categorical Fisher is `vp.ggnvp(model, vp.loss.softmax_cross_entropy(...))`, the same object by the math; `fisher_vp` stays a distinct family for the score-gradient outer-product construction. The five string knobs of the old Fisher constructors (`distribution`, `label_policy`, `sample_space`, `score_reduction`, `denominator`) are validated fields of the `Likelihood`. The opaque `representation` mapping of the old metric constructors is the typed `Metric`. Every closed-set field of these objects is validated at construction and raises on a typo.

`combine` is an operator expression over the named children. `vp.compose(e_1, ..., e_k)` is sequential application $e_1(e_2(\cdots e_k(v)))$, `vp.linear_combination((c_1, e_1), ..., (c_m, e_m))` is the weighted sum $\sum_i c_i e_i(v)$, `vp.scaled_identity(c)` is the leaf $c v$, and `vp.source(child)` is a leaf that seeds $v$ from a batch-to-vector child. `vp.compose` and `vp.linear_combination` close the linear operator algebra under composition, addition, and scalar multiplication; `vp.scaled_identity(c)` supplies the operator $cI$, making the algebra unital. `vp.source(child)` is a depth-0 leaf that seeds $v$ from a batch-to-vector child, so a source-bearing expression is vector-valued, a generator of $v$ rather than an operator on an external $v$. Preconditioning $M^{-1}H$ is `vp.compose("inverse_metric", "hvp")`, damping $H+\lambda I$ is `vp.linear_combination((1.0, "hvp"), (lam, vp.scaled_identity(1.0)))`, averaging is a `vp.linear_combination` over the terms, and the natural-gradient step $F^{-1}\nabla f$ is `vp.compose("inverse_fisher", vp.source("loss_gradient"))`. The child-name string leaves of the expression are the declared `children` and are the composition's dependencies; `vp.scaled_identity` is not a child; admission rejects an expression whose child-name string leaves are not exactly `children`. `vp.compose` requires each node's output space to match the next node's input space, and `vp.linear_combination` requires its terms to share input and output space.

The expression type is `Combine`:

```python
Leaf = str | ScaledIdentity | Source        # a child-name string, vp.scaled_identity(c), or vp.source(child)
Combine = Compose | LinearCombination | Leaf

def compose(*terms: Combine) -> Compose: ...                         # e_1(e_2(... e_k(v)))
def linear_combination(*weighted: tuple[float, Combine]) -> LinearCombination: ...   # sum_i c_i e_i(v)
def scaled_identity(c: float) -> ScaledIdentity: ...                 # c v
def source(child: str) -> Source: ...                               # seeds v from a batch-to-vector child
```

`vp.source(child)` seeds the threaded vector and so may appear only as the innermost argument of a `vp.compose` or as a term of a `vp.linear_combination`; admission rejects a source in any other position, where a later node would be applied to it as a function (`vp.compose(vp.source("g"), "hvp")` is rejected). An expression that contains a source is vector-valued and the composition is called as `composition(batch)`; a source-free expression is an operator called as `composition(batch, vector)`. Inversion stays with `inverse_metric_vp`, which solves $M^{-1}v$. A `vp.metric.matrix_free` metric is a PSD curvature, a GGN or Fisher, and a PSD curvature is generically singular, so `conjugate_gradient` requires the shifted operator $M+\lambda I$ with $\lambda>0$ to be positive-definite: $(G+\lambda I)^{-1}v$ is `inverse_metric_vp(vp.metric.matrix_free(operator=ggn), damping=lam)` with `lam > 0`. Admission rejects `damping=0` on a `conjugate_gradient`, `cholesky_solve`, `eigh_solve`, or `svd_solve` row over a PSD-but-not-PD metric, since a zero eigenvalue makes the inverse undefined.

A `Case` pairs one batch with one vector. It is the reference and probe input the tuner forms from the `data` and `vectors` iterables, and the type `vp.case(batch=..., vector=...)` builds for the `reference=` and `probes=` overrides. A `Run`, returned by `vp.tune(products=...)`, maps each product name to its tuned `Operator`:

```python
def case(*, batch: Batch | None = None, vector: TensorTree | None = None) -> Case: ...

class Case(Protocol):
    batch: Batch | None
    vector: TensorTree | None

class Run(Protocol):
    plan: Plan
    def __getitem__(self, name: str) -> Operator: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
```

The annotation types in these signatures are the typed objects the `vp.*` constructors produce, each validating its closed-set fields at construction: `Model` (`vp.torch_model`), `ParameterSurface` (`vp.parameters`), `ModuleCallSpec` (`vp.module_call`), `Output` (`vp.output`), `Loss` (`vp.loss.*`), `Likelihood` (`vp.likelihood.*`), `Metric` (`vp.metric.*`), `SampleSource` (`vp.samples.*`), `Damping` (`vp.damping.*`), `Target` (`vp.cuda`), `SearchSpace` (`vp.space.*`), `SearchStrategy` (`vp.search.*`), `Operator` (the operator constructors), and `Combine` with its node types `Compose`, `LinearCombination`, `ScaledIdentity`, and `Source` (`vp.compose`, `vp.linear_combination`, `vp.scaled_identity`, `vp.source`). Each is opaque to the caller; its fields are listed where the constructor is described.

The typed front-door objects build on lower-layer callable protocols that live in `vptune.ext`. `vp.loss.from_scalar(fn, ...)` wraps a `ScalarObjective`; `vp.output(...)` and the likelihood internals wrap a `FunctionObjective`. A normal caller uses the typed objects, not these protocols:

```python
class ScalarObjective(Protocol):
    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> torch.Tensor: ...

class FunctionObjective(Protocol):
    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> TensorTree: ...

class Materializer(Protocol):
    def identity(self) -> Mapping[str, Any]: ...

    def __call__(
        self,
        candidate: Candidate,
        record: FullSizeRecord,
    ) -> Any: ...

class PlanValidator(Protocol):
    def __call__(
        self,
        candidate: Candidate,
        record: FullSizeRecord,
        context: PlanValidationContext,
    ) -> ReferenceResult: ...
```

Extension API:

```python
import vptune.ext as vpx

candidate = vpx.Candidate(...)
runtime = vpx.standard_runtime_config(...)
operation_factory = vpx.standard_operation_factory(...)
reference_check = vpx.standard_reference_check(...)
manifest = vpx.axis_manifest()
record = vpx.plan_to_json(plan)
plan = vpx.plan_from_json(...)
current = vpx.plan_record_current(record, plan)
```

`vptune.ext` owns `Candidate`, `RuntimeConfig`, `OperatorSpec`, `Family`, the axis table and manifest, the package-owned anchors, candidate-axis builders, memory backends, checkpoint execution, Autobatch integration helpers, schema helpers, and standard runtime implementation types. These names are the adapter-author and custom-runtime API, not the root `vptune` namespace.

Mutation rules:

- Objective callables must treat `params`, `buffers`, `batch`, `vectors`, and `context` as read-only unless a candidate declares a mutation axis.
- A candidate that mutates module state must record the mutated state fields and restore them before the next probe.
- Probe output tensors must be detached or reduced to signatures before saved records are written.
- Declared mutable runtime state must be reset by the operation wrapper before every measured call.

Error types:

- `VPTuneError`
- `AdmissionError`
- `ReferenceFailedError`
- `NoPassedCandidateError`
- `StaleRecordError`
- `MeasurementError`
- `MaterializationError`

## Core Data Model

`Target` fields:

- Device list.
- Accelerator type.
- Allowed dtypes.
- Allowed core attention frontends and SDPA kernels.
- Timing policy.
- Selection policy.
- Determinism policy.
- Environment capture policy.

`vp.cuda(...)` constructs the core target. Adapter attention frontends and sharding allowances are carried by the adapter spaces composed into the search space, not by the core target.

`Problem` fields:

- Model.
- Parameter surface.
- Data iterable.
- Operator spec.
- Vector iterable.
- Target.
- Search space.
- Search strategy.
- Runtime config.
- Anchor policy.
- Replay policy.

A single-product `Problem` carries no cohort constraints; cohort coherence is a run-level
relationship and lives on the `TuningRun`.

`TuningRun` fields:

- Products. The operators tuned together; composition children are sibling products.
- Target.
- Search space.
- Search strategy.
- Cohort constraints. The `CohortConstraint`s the products must jointly satisfy.
- Data and vector iterables by product name.
- Problems, validators, and validator identities when an adapter lowers them.
- Run id.

`vp.cohort.layout_coherence(settings_keys)` builds a `CohortConstraint` whose
`settings_keys` are the listed axes and whose allowed assignments are the common values
those axes take across the search space; it pins, for example, `layout.vector`,
`layout.flatten_order`, and `dtype.vector` so the pilot's $H$, $R$, and $R^{-1}$ products
share one vector representation.

`RuntimeConfig` is a `vptune.ext` type. Its fields:

- Candidate rows.
- Operation factory.
- Reference check.
- Materializer.
- Materializer identity.
- Axis registry.
- Runtime identity signature.
- Autobatch domains for monotone integer candidate axes.

`Family` fields:

- Name.
- Operator spec.
- Dependencies. For composition families, dependencies are derived from the operator's ordered `children` field and are not separately declared.
- Candidate generator.
- Anchor checks.
- Full-size probe inputs.
- Materialization rule.

`Candidate` is a `vptune.ext` type. Its fields:

- Family.
- Row id.
- Axis ids changed by this row.
- Settings.
- Dependency identity records.
- Cohort assignment identity.
- Admission status.
- Admission failure fields.
- Candidate generator id.
- Candidate generator version.
- Source id for imported rows; package-generated rows set source id to `package`.

`Check` fields:

- Name.
- Input signature.
- Candidate settings.
- Measurements.
- Thresholds.
- Status.
- Error type and error message on failure.

`Measurement` fields:

- Elapsed seconds.
- Peak allocated MiB.
- Peak reserved MiB.
- Post allocated MiB.
- Post reserved MiB.
- Backend-specific device memory measurement fields.

`Plan` fields:

- Selected candidate per family.
- Selected settings per family.
- Candidate table.
- Check rows.
- Full-size rows.
- Selected-plan validation rows.
- Package version.
- PyTorch version.
- Model adapter version.
- Candidate generator versions.
- Environment signature.
- Materializer identities.
- Target identity.
- Runtime identities.
- Adapter identities.
- Whether selected-plan validation is required.
- Validator identities when validation is required.
- Selected cohort assignment.
- Cohort constraints.
- Dependency graph.

`ReplayContext` fields:

- Current run input signature.
- Current per-family input signatures.
- Current target identity.
- Current runtime identities.
- Current adapter identities.
- Current validator identities.
- Current materializer identities.
- Current selection policy.
- Whether selected-plan validation is required during replay.
- Expected validation order when validation is required.

`CohortConstraint` fields:

- Constraint name.
- Settings keys.
- Allowed assignments.
- Family coverage rule.
- Dependency inheritance rule. The supported value is `covered_families`.
- Selection aggregation rule. The supported value is `sum_median_elapsed_seconds`.

`AutobatchDomain` fields:

- Axis name for one integer domain.
- Ordered positive integer domain values.
- Explicit settings for every domain value. A value may set several runtime knobs when those knobs are semantically coupled by the generated implementation.
- Value-to-settings function id.
- Admission identity.
- Optimization goal.
- Timing policy.
- Device list.
- Reuse fields compared directly during replay.

`AxisDescriptor` fields:

- Axis key.
- Owner id.
- Allowed values or integer-domain id.
- Operators to which the key applies.
- Class A annotation when applicable.
- Class B annotation when applicable.
- Class C primary group.
- Class C merge rules.
- Admission rule id.
- Lowering rule id.
- Adapter id when an adapter owns lowering.
- Alias normalization rule.
- Required reference checks.
- Required full-size checks.
- Settings keys read for admission.
- Settings keys written by the axis.

`AxisManifest` fields:

- Package version.
- Manifest version.
- Axis descriptors.
- Complete Class C partition.
- Class C merge rules.
- Alias normalization rules.
- Integer-domain definitions.
- Adapter-owned axis registrations.
- Target allowance fields that filter admitted values.

`ExecutionLowering` fields:

- Family.
- Row id.
- Runtime owner id.
- Candidate settings.
- Normalized settings.
- Prepared parameter layout.
- Prepared vector layout.
- Prepared batch layout.
- Prepared compile boundary.
- Prepared attention frontend and kernel.
- Prepared distributed process and placement fields.
- Reference operation builder.
- Measured operation builder.
- Materializer builder.
- Required reference checks.
- Required full-size checks.

## Replay Fields

Python objects such as `model`, objective callables, data iterators, vector iterators, and include rules are runtime inputs. Saved records never pretend to serialize them. Each runtime input must provide explicit replay fields that describe the current computation.

Required replay fields:

- Model fields: class name, package version, config fields, parameter name list, parameter shape list, buffer name list, buffer shape list, tied-weight groups, active parametrizations, train/eval mode, device placement, dtype policy, source revision, parameter value version, buffer value version, and adapter id.
- Parameter surface fields: included parameter names, flattened order, shape list, trainable flags, tied-weight treatment, buffer policy, parametrization policy, parameter value version, and buffer value version.
- Loss, output, and likelihood fields: the typed object kind, the validated closed-set fields (reduction, denominator, sample space, label policy), the qualified callable name and the caller-declared value version when a custom `from_scalar`, `declared_psd_matrix_free`, or `output` callable is wrapped, RNG policy, module mode, and gradient target. Editing a wrapped callable without bumping its value version is a caller error that reuses stale numeric checks.
- Data fields: dataset name or user id, data value version, revision, selected row ids or slice fields, and batch collation policy.
- Vector fields: vector source id, vector value version, shape tree, dtype tree, norm summary, seed when generated, and storage id when loaded from tensors.
- Metric fields: metric kind, metric value version, declared factor names, factor order, factor shape tree, block order, denominator, and normalization fields; for a `matrix_free` metric, the wrapped operator's product name and its selected-plan identity stand in for the factor fields.
- Composition expression fields: the tree shape of the `Combine` expression, every `linear_combination` coefficient, every `scaled_identity` constant, every child-name leaf, every `source` child, and the `call_inputs` the expression implies.
- Damping fields: the damping kind (scalar, per-group, KFAC-pi) and its values.
- Solver-tolerance fields: the residual tolerance `tol` on the iterative inverse and inverse-inner rows, setting the iterative stopping criterion and the accepted residual.
- Sample source fields: the sample source kind (fixed_seed, table), the seed when fixed_seed, the sample count, the table identity when table, and the sampling-bound formula id when exact-Fisher comparison is enabled.
- Bound-operator fields: whether the operator is batch-bound (`is_bound`), and the bound batch signature (shape tree, dtype tree, value version). A bound operator compiles the `bound_operator_vector_step` boundary and scores its call horizon over the fixed-batch inner loop; `bind` on a data-independent operator is rejected.
- Inner-product fields: the norm declaration `as_norm` for `metric_inner_vp` and `inverse_metric_inner_vp`.
- Target fields: declared devices, per-device hardware signatures, GPU model, device capability, driver version when CUDA reports it, CUDA or ROCm runtime version, PyTorch version, `torch.__config__` summary, allocator config, deterministic flags, TF32 flags, cuDNN flags, BF16 reduced-reduction flags, matmul precision, MPS availability, and relevant environment variables. `vptune.ext.environment_signature()` captures the runtime fields, and `Target.signature()` binds both the declared target devices and their device signatures. The selected memory backend fields are part of the per-problem input signature because callers may supply the backend at tune time.
- Adapter fields: adapter package version, adapter registry id, model-specific admission rules, and candidate-generator version.

Every saved row is current only when its saved replay fields match the current run by direct field equality: family, row id, check name for reference rows, input signature, settings, thresholds, dependency fields, cohort assignment fields, changed axes for candidate rows, generator fields, axis descriptor fields, admission fields, migration source id, and selected dependency fields.

Value-version fields are caller-declared drift signals. The package compares those fields directly and does not inspect tensor content during replay. Callers must change parameter, buffer, data, vector, metric, wrapped-callable (`from_scalar`, `declared_psd_matrix_free`, `output`), or sample-source value-version fields when value changes should invalidate saved numeric checks or selected rows.

When runtime fields declare `adapter_id` and `adapter_version`, the enclosing `Problem.adapter_identity` must declare the same values. Mismatched adapter fields fail before search.

## Operator Semantics

`vp.gradient(model, loss)` declares $\nabla_\theta f(\theta)$ for the model's parameter surface, called as `gradient(batch)`. The loss carries $f$.

`vp.jvp(model, output)` declares $J_f(\theta)v$, called as `jvp(batch, vector)`. The output carries the vector function $z(\theta)$.

`vp.vjp(model, output)` declares $J_f(\theta)^\top u$, called as `vjp(batch, cotangent)`.

`vp.hvp(model, loss)` declares $\nabla_\theta^2 f(\theta)v$, called as `hvp(batch, vector)`. The Hessian is indefinite; among the parameter-space curvature operators (`hvp`, `ggnvp`, the Fishers, `metric_vp`), `hvp` is the only one that may be indefinite, and `jvp`/`vjp` map between distinct tangent and cotangent spaces where the term does not apply.

`vp.ggnvp(model, loss)` declares $J^\top H_\ell Jv$ for output Jacobian $J=\partial z/\partial\theta$ and output-space loss Hessian $H_\ell=\partial^2\ell/\partial z^2$, called as `ggnvp(batch, vector)`. Generalized Gauss-Newton is defined for a loss convex in the output, so $H_\ell\succeq 0$ and $G\succeq 0$, and PSD is intrinsic to the object. The loss carries $H_\ell$, and `ggnvp` enforces symmetry, PSD, and the dot-product identity on it in two stages: a closed-form loss with known PSD status (`softmax_cross_entropy`, `kl`, `mse`) is decided at admission; `vp.loss.from_scalar` passes admission and its $H_\ell$ is checked by the symmetry and PSD reference checks at the probe, which certify point PSD only, so the caller owns output convexity off the probe. `vp.loss.declared_psd(factors=...)` is PSD by construction; `vp.loss.declared_psd_matrix_free(matvec=...)` is checked by a Lanczos eigenvalue estimate with a declared iteration count and a residual bound, admitted only when $\theta_1-\|r_1\|\ge-\tau$ for the smallest Ritz value $\theta_1$ and its residual norm $\|r_1\|$ (the smallest Ritz value alone is an upper bound on $\lambda_{\min}$, so the residual correction is required), or refused at admission.

`vp.fisher_vp(model, likelihood)` declares $Fv = \mathbb{E}[s_\theta s_\theta^\top v]$, where $s_\theta=\nabla_\theta \log p_\theta(y|x)$ is the score the likelihood defines, called as `fisher_vp(batch, vector)`. FisherVP rows compute exact score-gradient outer products over the declared score source. Exact categorical NLL Fisher is represented by GGNVP with the CE or KL loss Hessian; the two are the same object when the loss `reduction` and the likelihood `denominator` apply the same per-token (or per-example) normalization over the same mask, which the typed objects make explicit.

`vp.sampled_fisher_vp(model, likelihood, samples=...)` declares $\hat F_S v = \frac{1}{nS}\sum_{i,s} g_{is}(g_{is}^\top v)$ for the sample source the `samples` argument declares, called as `sampled_fisher_vp(batch, vector)`. It is a separate operator family from exact FisherVP. The sample source carries the fixed sample table or the fixed seed and sample count $S$, and the exact-Fisher comparison formula used when a row enables exact-Fisher comparison. That formula is a declared known bound id (a matrix-Bernstein or Hutchinson relative-variance bound as a function of $S$ and the score moments) that the package interprets; a free-form bound is refused at admission.

`vp.empirical_fisher_vp(model, loss)` declares $\frac{1}{n}\sum_i g_i(g_i^\top v)$ for per-example gradients $g_i = \nabla_\theta \ell_i(\theta)$, called as `empirical_fisher_vp(batch, vector)`. The loss carries the within-example loss reduction and denominator. The standard anchor computes those gradients from the declared loss and data axis. A supplied `per_example_gradients` matrix is a dense candidate input, not the semantic anchor.

`vp.metric_vp(model, metric)` declares $Mv$. The operator declares its `call_inputs`: a declared, factored metric is data-independent and is called as `metric_vp(vector)`; a `vp.metric.matrix_free(...)` metric whose forward operator is data-dependent is called as `metric_vp(batch, vector)`. The arity is the operator's declared `call_inputs`, inspectable on the operator, not a hidden mode. The metric object returns multiply, inverse multiply, inner product, and factor records when it declares factors. It is one of dense matrix, diagonal tree, block-diagonal blocks, KFAC factors, eigenvalue-corrected KFAC (EKFAC) factors, low-rank factors, GGN-derived factors, or a matrix-free PSD forward operator. The metric fields are fixed problem fields, not sweep axes.

`vp.metric_inner_vp(model, metric, as_norm=...)` declares the Gram $U^\top M V$ for stacked vectors $U,V$ each $n\times k$; the $k=1$ case is the scalar $u^\top M v$. Its `call_inputs` follow the metric like `metric_vp`: `metric_inner_vp(left, right)` for a declared, factored metric and `metric_inner_vp(batch, left, right)` for a `matrix_free` metric. The reduction path is a sweep axis: `multiply_then_reduce` applies $MV$ and forms $U^\top(MV)$, `factored_gram` forms the $k\times k$ Gram directly from the metric factors in a Kronecker-aware order, and `sqrt_apply_reduce` applies the square-root factor adjoint $L^\top$ (the factor `sqrt_metric_vp` builds, $LL^\top=M$) to $U$ and $V$ and forms $(L^\top U)^\top(L^\top V)$, so the diagonal is $\lVert L^\top v\rVert^2\ge 0$ by construction. A generalized eigensolver reads that diagonal as the squared $M$-norm for $M$-orthonormalization, and `as_norm=True` declares that use: it admits only `sqrt_apply_reduce`, the one path whose diagonal is exactly nonnegative for every input, including the near-null-space vectors a probe sweep never reaches. `as_norm` is a fixed problem field, not a sweep axis.

`vp.inverse_metric_vp(model, metric, damping=..., tol=...)` declares $(M+\lambda I)^{-1}v$ for the declared damping. Its `call_inputs` follow the metric the same way `metric_vp` does. A `vp.metric.matrix_free` metric is a PSD curvature inverted only through the iterative solve paths; a PSD curvature is generically singular, so `conjugate_gradient` requires $\lambda>0$, and $(G+\lambda I)^{-1}$ on a matrix-free GGN or Fisher is `inverse_metric_vp(vp.metric.matrix_free(operator=ggn), damping=lam)` with `lam > 0`. `damping` is a typed `Damping` (scalar, per-group, or KFAC-pi factor split), and an iterative row carries a residual tolerance `tol` that sets both the CG stopping criterion and the accepted inverse residual.

`vp.inverse_metric_inner_vp(model, metric, damping=..., as_norm=..., tol=...)` declares the Gram $U^\top(M+\lambda I)^{-1}V$ for the declared damping, with $U,V$ each $n\times k$. Its `call_inputs` follow the metric like `inverse_metric_vp`. The reduction path is a sweep axis: `solve_then_reduce` solves $(M+\lambda I)X=V$ and forms $U^\top X$, `factored_gram` forms the Gram from the inverse factors (EKFAC through the corrected eigenvalues in the Kronecker eigenbasis), and `sqrt_apply_reduce` applies the inverse-square-root factor adjoint $L^\top$ (the factor `inverse_sqrt_metric_vp` builds, $LL^\top=(M+\lambda I)^{-1}$) to $U$ and $V$ and forms $(L^\top U)^\top(L^\top V)$. That last path is a forward factor application and not a solve, so the diagonal is $\lVert L^\top r\rVert^2\ge 0$ even under the matrix-free Lanczos square root, whose value stays nonnegative regardless of approximation quality. A generalized eigensolver reads that diagonal as the squared $R^{-1}$-norm of its residual, and `as_norm=True` admits only `sqrt_apply_reduce`. The positive-damping requirement on a PSD-but-singular metric is the same as `inverse_metric_vp`, and the operator carries the residual tolerance `tol`.

`vp.sqrt_metric_vp(model, metric)` and `vp.inverse_sqrt_metric_vp(model, metric, damping=..., tol=...)` apply a factor $Lv$ and its adjoint $L^\top v$, with $LL^\top=M$ for the square root and $LL^\top=(M+\lambda I)^{-1}$ for the inverse square root; the inverse-square-root operator applies $Lv$ for the damped-inverse factor, not $L^{-1}v$. The $Lv$ application is what a weight-space posterior sample $\theta=\mu+L z$ with $z\sim\mathcal{N}(0,I)$ needs, and the adjoint $L^\top v$ is what the metric inner product's `sqrt_apply_reduce` path uses for the $M$-norm and $R^{-1}$-norm; the result is a valid covariance factor, not the symmetric square root unless an eigenbasis path is selected. The forward factor per metric kind, with $LL^\top=M$: diagonal uses the pointwise square root; KFAC and EKFAC use the closed-form factor square root in the Kronecker eigenbasis; low-rank $M=UU^\top+D$ uses $L=[U, D^{1/2}]$; GGN-derived $M=J^\top H_\ell J$ uses $L=J^\top H_\ell^{1/2}$; dense and block use a Cholesky factor; a matrix-free metric uses a Lanczos approximation of $f(M)v$ with $f(t)=t^{1/2}$. The damped inverse factor, with $LL^\top=(M+\lambda I)^{-1}$, is direct for the kinds whose damped inverse stays elementwise or in a known eigenbasis: diagonal inverts $d+\lambda$, KFAC and EKFAC inverse-square-root the Kronecker spectrum shifted by the declared damping (the eigenbasis for KFAC scalar or per-group damping, the factored shift for `kfac_pi`, the corrected-eigenvalue floor for EKFAC), and dense and block take a Cholesky of the dense damped inverse. Low-rank and GGN-derived build the inverse factor through the Woodbury capacitance matrix, and a matrix-free metric uses a Lanczos approximation with $f(t)=(t+\lambda)^{-1/2}$.

`vp.per_example_gradient(model, loss)` declares the stacked per-example gradients $\{g_i\}_i$ with $g_i=\nabla_\theta\ell_i(\theta)$, called as `per_example_gradient(batch)`. Its output is a fixed problem field: a parameter tree whose every leaf gains a leading axis of size $n$, the batch's example count taken from the declared data axis, so a consumer can bind to the shape. It is the object influence functions and per-group unlearning consume directly, with the per-example gradient paths `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, and `backward_materialized_grad`; empirical Fisher is `per_example_gradient` followed by an outer-product reduction.

`vp.composition(model, children=..., combine=...)` declares a composition of selected operator implementations. `children` is the child-product list and is the single source for the composition's dependencies; `combine` is the operator expression that arranges them. The built-in composition runtime paths are `sequential_composition`, which applies a `vp.compose` node's components in declared order to the current vector, and `linear_combination`, which applies a `vp.linear_combination` node's terms to the same vector and reduces them with the declared coefficients.

Every operator spec declares:

- Parameter surface.
- Data axis.
- Aggregation rule, carried by the typed object (the loss reduction, the likelihood denominator and sample space, or the sample source), not a separate operator argument.
- Output shape.
- Dtype policy.
- Required batch inputs by phase: `reference` and `operation`.
- Randomness policy.
- Anchor family.
- Numeric thresholds.

All built-in operators obey these rules:

- Operation and reference entry points validate declared batch inputs before objective execution.
- `vp.hvp(...)` declares `symmetry_vector` for reference checks.
- `vp.vjp(...)` declares `tangent_vector` for reference checks.
- `vp.ggnvp(...)` declares `symmetry_vector` for the dot-product reference check. The output-space Hessian $H_\ell$ is produced by the typed loss, not a caller-supplied batch input; the PSD and symmetry reference checks run on it.
- `vp.metric_vp(...)` and `vp.inverse_metric_vp(...)` declare the fields needed by the selected metric.
- Fisher, sampled Fisher, and empirical Fisher dense candidate paths add dense matrix inputs to the declared base inputs: `score_gradients` for `dense_score_outer`, `sampled_score_gradients` for sampled Fisher dense rows, `per_example_gradients` for `dense_empirical_fisher`, and denominator inputs such as `normalization` or `num_examples` when their declared denominator needs a batch field.
- Metric representations add their declared inputs: `metric_matrix` for dense, `metric_diagonal` for diagonal, `metric_blocks` for block diagonal, `kfac_factors` for KFAC, `ekfac_eigvecs_a`/`ekfac_eigvecs_g`/`ekfac_corrected_eigenvalues` for EKFAC, `low_rank_factors` for low rank, and `ggn_factors` for GGN-derived factors. A matrix-free metric adds the wrapped product's name instead of factors.
- Flattening order is the declared parameter-surface order. The order is stored in the parameter surface identity and compared directly during replay.
- Dense GGNVP, metric, inverse metric, FisherVP, sampled FisherVP, and empirical FisherVP flatten full vector trees for matrix multiplication and reconstruct the original tree shape on return.
- Tree outputs preserve key order. Vectorized outputs declare whether vectors are stacked on a leading axis or returned as a sequence.
- The math aggregation is carried by the typed object: the loss `reduction` (`sum`, `mean`, `token_mean`, or a named adapter reduction) for loss-based operators, the likelihood denominator and sample space for exact Fisher, and the sample source for sampled Fisher. There is no separate `aggregation` argument, for the same reason there is no `loss_geometry`: a string that changes the math belongs on a typed object.
- Segmented sum, segmented mean, and microbatch accumulation are execution rows (`batch.data_microbatch_size`, `schedule.gradient_accumulation`) that preserve the declared aggregation, not semantic choices. Per-example structure is intrinsic to `empirical_fisher_vp`, not a reduction of `gradient`.
- `None` gradients are represented as zero tensors with matching parameter shape only when the parameter is declared active and mathematically disconnected.
- A parameter excluded from the parameter surface never appears in a returned vector tree.
- Buffers are part of the functional input unless the parameter surface declares them fixed.
- Tied weights are preserved unless the operator spec declares independent tied leaves.
- Module mode is explicit. Dropout and other stochastic layers are disabled by default for reference checks unless the randomness policy says otherwise.
- Randomness policy records seeds, `torch.Generator` state, `vmap` randomness mode, and checkpoint RNG handling.
- Adapter-specific inputs such as token weights and masks are part of the adapter identity.
- Relative error uses the declared denominator. When the denominator is zero, the row must pass the absolute threshold.

## Built-in Anchors

`vptune` owns anchors for standard operators.

Gradient anchors:

- `torch.func.grad` on a pure functional scalar.
- `torch.autograd.grad` on a small eager scalar.
- Finite-difference directional check.
- Segmentation invariance check for chunked data.

JVP anchors:

- `torch.func.jvp` for pure functions.
- `torch.autograd.forward_ad` for module-local tangent injection.
- Finite-difference directional output check on small inputs.

VJP anchors:

- `torch.func.vjp` for pure functions.
- `torch.autograd.grad` with explicit `grad_outputs` on eager functions.
- Dot-product identity check: $\langle Jv,u\rangle = \langle v,J^\top u\rangle$. The tangent vector for this check is an explicit reference-batch field. The reference tangent and cotangent must satisfy $\|v\| \ge \tau$ and $\|u\| \ge \tau$ for a fixed positive $\tau$; admission rejects a zero or near-zero probe, which passes the identity vacuously.

HVP anchors:

- Reverse-over-reverse through `torch.autograd.grad`.
- `torch.autograd.functional.hvp` for small scalar references.
- `torch.autograd.functional.vhp` for small scalar references when symmetry and smoothness checks pass.
- `torch.func.jvp(torch.func.grad(f))` for pure functions with forward AD coverage.
- Symmetry check: $\langle x,Hy\rangle = \langle y,Hx\rangle$. The reference vectors $x$ and $y$ must satisfy $\|x\| \ge \tau$ and $\|y\| \ge \tau$ for a fixed positive $\tau$; admission rejects zero or near-zero probes, which pass the identity vacuously.
- Finite-difference gradient-direction check.
- Segmentation invariance check.

GGNVP anchors:

- Explicit JVP through model outputs, exact loss-Hessian product in output space, and VJP back to parameters.
- Dense output-Jacobian construction on small references.
- Dot-product identity check on the output-space loss Hessian.
- Loss-Hessian shape check against flattened model output.
- Finite-value check for the output metric and candidate product.
- Symmetry and PSD checks on the output-space loss Hessian.
- Cross-check between dense-Jacobian product and JVP-Hessian-VJP product.

FisherVP anchors:

- Exact score-gradient outer products on small references from the declared distribution.
- Dense Fisher matrix on tiny models.
- Explicit-score Fisher rows require `score_reduction="none"`. Precomputed `score_gradients` matrices are admitted as dense candidate inputs with their own identity and normalization fields.

Sampled FisherVP anchors:

- Fixed sample table or fixed seed and sample count.
- Explicit sampled score-gradient outer products on small references.
- Dense sampled Fisher matrix on tiny models.
- Exact-Fisher comparison only when the operator spec declares the sampling-bound formula.
- Precomputed `sampled_score_gradients` matrices are admitted as dense candidate inputs with their own identity and normalization fields.

EmpiricalFisherVP anchors:

- Per-example gradients by for-loop from the declared per-example loss.
- Per-example gradients by `vmap(grad)` where function purity permits it.
- Dense empirical Fisher matrix on tiny models.
- Precomputed `per_example_gradients` matrices are admitted as dense candidate inputs with their own identity and normalization fields.

PerExampleGradient anchors:

- Per-example gradients by for-loop from the declared per-example loss, stacked on a leading axis.
- Per-example gradients by `vmap(grad)` where function purity permits it.
- Agreement that the outer-product reduction of the stacked gradients equals the empirical FisherVP anchor.

MetricSquareRoot anchors:

- Dense factor check on small references: the applied factor satisfies $LL^\top=M$ for the square root and $LL^\top=(M+\lambda I)^{-1}$ for the inverse square root.
- Eigenbasis cross-check for KFAC and EKFAC; Cholesky cross-check for dense and block metrics.
- Repeated-draw covariance check for the matrix-free Lanczos path.

Metric anchors:

- Dense matrix multiply, solve, and inner product on small block references.
- Dense references are reconstructed from the declared representation before the check: dense matrix uses $M$ directly; diagonal tree assembles $\operatorname{diag}(d)$ in parameter order; block-diagonal blocks assemble $M=\operatorname{blockdiag}(M_1,\ldots,M_b)$ in declared block order; KFAC assembles each block as $A_b \otimes G_b$; EKFAC assembles each block as $(U_{a,b}\otimes U_{g,b})\operatorname{diag}(s_b)(U_{a,b}\otimes U_{g,b})^\top$ from the eigenbases and corrected eigenvalues; low-rank assembles $M=UU^\top + D$; GGN-derived assembles $M=J^\top H J$.
- Metric multiply, inner product, and inverse references use the reconstructed dense $M$ plus the declared damping when present.
- Inverse residual check: $\|(M+\lambda I)x-v\|/\|v\|$ for damped inverse rows and $\|Mx-v\|/\|v\|$ for undamped inverse rows.
- Symmetry check.
- PSD check by eigenvalue floor for dense references.
- A matrix-free metric is checked through its operator's own anchors for the forward multiply and through the inverse residual for the conjugate-gradient solve, not through dense reconstruction.
- Positive damping and conditioning checks for inverse rows when the candidate declares a damped metric.

Metric inner-product anchors:

- Gram against the reconstructed dense reference: $U^\top M V$ for the forward inner product and $U^\top(M+\lambda I)^{-1}V$ for the inverse inner product, over small block references.
- Diagonal nonnegativity on the same-vector entries ($U=V$) for an `as_norm` row; the off-diagonal entries compare against the dense reference only, since a general bilinear entry has no sign to check.
- For `sqrt_apply_reduce`, agreement that the Gram equals $(L^\top U)^\top(L^\top V)$ formed from the square-root anchor's factor.
- Inverse residual on each solved column for an `inverse_metric_inner_vp` `solve_then_reduce` row, and the same positive-damping requirement on a PSD-but-singular metric.

Low-precision candidate rows compare against anchors run without candidate dtype downcasting. Rows that degrade reduction precision must also pass the derived numeric error bound from `FEATURES.md`.

Composite anchors:

- Run each child operator's anchor.
- Save child reference rows before the parent composition reference row.
- Run the candidate component chain on a small deterministic input.
- Run the anchor component chain on the same input.
- Compare candidate and anchor outputs after every component.
- Fold the worst component error into the composition's thresholded `max_abs_diff` and `max_rel_diff`.
- Check output agreement with dense composition where dimensions permit it.
- The component order, candidate component identities, anchor component identities, child names, and ordered child reference descriptors are explicit runtime identity or measurement fields.

## Candidate Axes

`vptune` provides a feature manifest, exposed as `vptune.ext.axis_manifest()`. The manifest is the union of the core registry and the adapter registrations, so it contains every axis key in `FEATURES.md` with exactly one owner each. The core registry owns only the axes the core runtime lowers. The adapter attention frontends (`transformers_*`, `paged|*`, `registered_transformers_attention`) and the whole distributed family (`distributed.*`, `dtensor.*`, `fsdp.*`, `tp.*`, `sequence_parallel.*`, `context_parallel.*`, `comm.*`) are owned by the Transformers and distributed adapter registries and composed into the search space through `space.with_attention(...)` and `space.with_distributed(...)`. Axis registration is admission; execution belongs to the runtime that owns the candidate.

Every axis descriptor uses the `AxisDescriptor` fields from the core data model. There is one descriptor shape in the package.

The manifest must cover these sections from `FEATURES.md`:

- vectorization
- gradient materialization
- model call and functionalization
- attention execution
- batching, chunking, and input representation
- activation and memory schedule
- dtype and numeric backend
- torch compile
- kernel fusion
- parameter and vector layout
- distributed execution
- metric storage
- inverse solve
- composition execution

Every integer-valued axis must be finite before candidate generation. It has either an explicit positive integer tuple or an `AutobatchDomain` with min value, max value, initial value, growth rule, objective, failure signals, termination rule, and value-to-settings mapping.

The package-owned standard runtime executes these settings:

- operator paths for gradient, JVP, VJP, HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, per-example gradient, metric multiply, metric inner product, metric square-root multiply, inverse metric multiply, inverse metric inner product, inverse metric square-root multiply, and composition
- dtype fields, autocast fields, matmul precision fields, and reduced-precision-reduction fields
- vectorization fields for vector, tangent, and cotangent batching
- `vmap_chunk_size` for vmap-owned paths
- compile fields when the callable boundary is package-owned
- dense metric and inverse-metric storage fields
- direct tensor-tree layout fields that do not require DTensor

The standard runtime rejects attention frontend settings, packed-token attention, checkpoint execution, DTensor layout, sharding, process-group communication, tokenizer-aware batching, and model-library-specific kernels unless the package owns an executor for the exact key. Adapter runtimes execute those settings through their own operation factories and admission rules. Composition uses `composition_runtime_config`, which owns the package composition paths and child-anchor checks.

`empirical_fisher.grad_path=vmap_grad` belongs to EmpiricalFisherVP. It uses `torch.func.vmap(torch.func.grad(...))`, maps every tensor in the batch over its leading axis, and passes non-tensor batch fields unchanged. All mapped tensors must have the same nonzero leading size. The per-example objective must return exactly one scalar for each mapped example. FisherVP uses exact score-gradient rows or precomputed score-gradient matrices. Sampled FisherVP uses fixed sampled score-gradient rows or precomputed sampled-score matrices.

For checkpoint rows, adapters call `checkpoint_operation(candidate, function, args, policy_key=...)`. The helper maps declared disabled policies to direct execution, validates active checkpoint fields through `admit_checkpoint`, and executes active rows with PyTorch non-reentrant checkpointing, declared RNG preservation, declared determinism checks, declared context function, and declared early-stop behavior.

Each axis is orthogonal. Coupled settings are represented as one named composite axis, such as `hvp_profile`, with every member written into the candidate row.

Adapter-defined axes must be registered through an adapter axis descriptor with:

- Axis name.
- Allowed values.
- Settings keys owned by the axis.
- Admission setting keys read by the axis. These keys are allowed by the registry without making them required axis values.
- Admission rule.
- Failure reason fields.
- Adapter id.
- Adapter version.
- Tests that exercise accepted and rejected values.

`vptune.ext.standard_axis_registry(exclude=(...))` builds a core registry with selected core axes omitted so an adapter can replace an owned setting key, such as `attention.frontend`, with adapter-specific admission and values. Adapter registries also omit core admission axes whose owned keys are only adapter admission fields for that model.

Candidate grid generation is registry-aware. For a multi-key axis, each grid value is a mapping whose keys exactly match the axis-owned setting keys.

### Axis Manifest Contents

The enumeration below is the full manifest, the union of the core registry and the adapter registrations. The adapter attention frontends (`transformers_*`, `paged|*`, `registered_transformers_attention`) and the distributed family (`distributed.*`, `dtensor.*`, `fsdp.*`, `tp.*`, `sequence_parallel.*`, `context_parallel.*`, `comm.*`) are owned by the Transformers and distributed adapter registries; the core registry owns the rest, including the core attention executor's `attention.sdpa_kernel`, `attention.partition`, `attention.padding`, and the four model-agnostic `attention.frontend` values (`pytorch_sdpa_direct`, `patched_eager`, `packed_exact`, `blockwise_exact`).

The package manifest must include these operator-owned axes:

- `gradient.path`: `torch_autograd_grad`, `torch_func_grad`, `torch_func_grad_and_value`, `backward_materialized_grad`.
- `gradient.value_reuse`: `gradient_only`, `gradient_and_primal_value`.
- `gradient.graph_schedule`: `build_once`, `rebuild_per_call`.
- `jvp.path`: `torch_func_jvp`, `forward_ad_dual`, `torch_func_linearize`.
- `jvp.linearize_reuse`: `none`, `reuse_at_same_primal`.
- `vjp.path`: `torch_func_vjp`, `autograd_grad_outputs`, `backward_materialized_grad`.
- `vjp.closure_reuse`: `none`, `reuse_vjp_closure_at_same_primal`.
- `hvp.path`: `reverse_over_reverse`, `jvp_grad`, `autograd_functional_hvp`, `autograd_functional_vhp`, `forward_ad_dual`, `linearize_grad`.
- `hvp.graph_schedule`: `retain_graph_across_vectors`, `rebuild_graph_per_vector`.
- `hvp.primal_reuse`: `reuse_primal`, `recompute_primal`.
- `hvp.gradient_reuse`: `reuse_gradient_closure`, `recompute_gradient`.
- `ggn.jvp_path`: `torch_func_jvp`, `forward_ad_dual`, `torch_func_linearize`.
- `ggn.loss_hessian_path`: `closed_form_softmax_ce_kl`, `autodiff_loss_hvp`.
- `ggn.loss_hessian_kernel`: `dense_global`, `streaming_global`, `two_pass_chunked_global`.
- `ggn.vjp_path`: `torch_func_vjp`, `autograd_grad_outputs`.
- `ggn.jvp_reuse`: `reuse_jvp`, `recompute_jvp`.
- `ggn.cotangent_reuse`: `reuse_output_cotangent`, `recompute_output_cotangent`.
- `fisher.expectation_path`: `explicit_full_expectation_score_rows`.
- `fisher.score_grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`.
- `fisher.accumulation`: `streaming_dot_accumulate`, `materialize_score_gradients`, `blockwise_score_matrix`.
- `sampled_fisher.sample_source`: `fixed_sample_table`, `fixed_seed_and_count`.
- `sampled_fisher.score_grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`.
- `sampled_fisher.accumulation`: `streaming_dot_accumulate`, `materialize_score_gradients`, `blockwise_score_matrix`.
- `sampled_fisher.exact_fisher_check`: `disabled`, `enabled_with_sampling_bound`.
- `empirical_fisher.grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`.
- `empirical_fisher.accumulation`: `streaming_dot_accumulate`, `materialize_per_example_gradients`, `blockwise_gradient_matrix`.
- `metric.multiply_path`: `dense_matmul`, `factorized_multiply`, `blockwise_multiply`, `streaming_multiply`.
- `metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`.
- `metric.accumulation`: `streaming`, `materialized_blocks`.
- `metric_inner.reduction_path`: `multiply_then_reduce`, `factored_gram`, `sqrt_apply_reduce`.
- `metric_inner.multi_rhs`: `single_column`, `block`.
- `inverse_metric.solve_path`: `dense_solve`, `cholesky_solve`, `eigh_solve`, `svd_solve`, `conjugate_gradient`, `factorized_solve`, `blockwise_solve`, `woodbury_low_rank_solve`.
- `inverse_metric.preconditioner`: `none`, `diagonal`, `block_diagonal`, `factorized_metric`, `matrix_free`.
- `inverse_metric.iteration_budget`: a finite positive integer domain.
- `inverse_metric.factor_reuse`: `refactor_each_rhs`, `reuse_factor_across_rhs`.
- `inverse_metric.block_schedule`: `layer_blocks`, `module_blocks`, `custom_blocks`.
- `inverse_metric.multi_rhs`: `single_column`, `block`.
- `inverse_metric_inner.reduction_path`: `solve_then_reduce`, `factored_gram`, `sqrt_apply_reduce`.
- `inverse_metric_inner.multi_rhs`: `single_column`, `block`.
- `sqrt_metric.factor_path`: `closed_form_factor_square_root`, `cholesky_factor`, `eigenbasis_factor`, `matrix_free_lanczos`.
- `sqrt_metric.lanczos_iterations`: a finite positive integer domain.
- `per_example_gradient.grad_path`: `torch_autograd_grad_loop`, `torch_func_grad`, `vmap_grad`, `backward_materialized_grad`.
- `per_example_gradient.accumulation`: `stacked_leading_axis`, `blockwise_stacked`.
- `composition.execution`: `materialize_each_child`, `stream_child_outputs`, `fuse_adjacent_children`, `compile_whole_composition`.
- `composition.child_evaluation`: `selected_child_rows`, `inline_child_lowering`.
- `composition.validation`: `validate_each_child`, `validate_composed_output`.

The package manifest must include these shared axes:

- `vectorization.mode`: `single_loop`, `manual_batch`, `vmap`.
- `vectorization.batch_size`: a finite positive integer domain.
- `vectorization.vmap_chunk_size`: a finite positive integer domain.
- `vectorization.in_dims`: declared PyTorch `vmap` input dimensions.
- `vectorization.randomness`: `error`, `same`, `different`.
- `call.path`: `functional_call`, `stateful_module`.
- `call.params`: `explicit_params`, `module_params`.
- `call.buffers`: `explicit_buffers`, `module_buffers`.
- `call.tied_weights`: `preserve_alias_groups`.
- `call.parametrizations`: `preserve_parametrizations`.
- `call.buffer_mutation`: `forbidden`, `declared_and_restored`.
- `call.grad_mode`: `grad_enabled`.
- `call.return_type`: `raw_tensor_tree`, `model_output_object_with_declared_fields`.
- `attention.frontend`: `transformers_eager`, `transformers_sdpa`, `transformers_flash_attention_2`, `transformers_flash_attention_3`, `transformers_flash_attention_4`, `transformers_flex_attention`, `paged|eager`, `paged|sdpa`, `paged|flash_attention_2`, `paged|flash_attention_3`, `paged|flash_attention_4`, `registered_transformers_attention`, `pytorch_sdpa_direct`, `patched_eager`, `packed_exact`, `blockwise_exact`.
- `attention.sdpa_kernel`: `math`, `flash_attention`, `efficient_attention`, `cudnn_attention`, `overrideable`, `priority_list`.
- `attention.custom_kernel_id`: a registered attention implementation id.
- `attention.mask_formatter_id`: a registered mask formatter id.
- `attention.partition`: `full`, `packed_tokens`, `blockwise_queries`, `segmented_forward_ad`.
- `attention.padding`: `dense_padded`, `unpadded_packed`.
- `batch.data_microbatch_size`: a finite positive integer domain.
- `batch.hvp_row_batch_size`: a finite positive integer domain.
- `batch.ggn_batch_size`: a finite positive integer domain.
- `batch.fisher_sample_batch_size`: a finite positive integer domain.
- `batch.empirical_example_batch_size`: a finite positive integer domain.
- `chunk.token_block_size`: a finite positive integer domain.
- `chunk.sequence_position_block_size`: a finite positive integer domain.
- `chunk.class_block_size_with_exact_global_normalization`: a finite positive integer domain.
- `chunk.output_cotangent_block_size`: a finite positive integer domain.
- `chunk.parameter_block_size`: a finite positive integer domain.
- `chunk.layer_block_size`: a finite positive integer domain.
- `chunk.lm_head_weight_chunk_bytes`: a finite positive integer domain.
- `schedule.per_example`: `loop`, `vmap`, `manual_batch`.
- `schedule.per_token`: `loop`, `packed`.
- `schedule.gradient_accumulation`: `single_step`, `microbatch_accumulate`.
- `input.batch_layout`: `dense_padded`, `packed_with_inverse_permutation`, `variable_length`.
- `input.length_grouping`: `none`, `exact_length_bucket`.
- `input.host_to_device`: `outside_measured_call`, `inside_measured_call`.
- `input.residency`: `cpu_staged`, `cpu_pinned`, `gpu`.
- `teacher_outputs`: `precomputed_cpu`, `precomputed_cpu_pinned`, `precomputed_gpu`, `recomputed_with_equality_check`.
- `checkpoint.use_reentrant`: `false`.
- `checkpoint.early_stop`: `false`, `true`.
- `checkpoint.preserve_rng_state`: `false`, `true`, under fixed RNG semantics.
- `checkpoint.determinism_check`: `default`, `none`.
- `checkpoint.context_fn`: `none`, `declared_context_pair`.
- `memory.primal_outputs`: `retain`, `recompute`.
- `memory.jvp_outputs`: `retain`, `recompute`.
- `memory.output_cotangents`: `retain`, `recompute`.
- `activation.recompute`: `none`, `checkpoint_non_reentrant_by_layer`, `checkpoint_selective`, `manual_recompute`.
- `activation.offload`: `none`, `saved_tensor_hooks_cpu`, `custom_saved_tensor_hooks`.
- `memory.vector_residency`: `gpu`, `cpu_pinned`, `cpu_staged`, `mmap_cpu`.
- `memory.intermediate_residency`: `gpu`, `cpu_pinned`, `cpu_staged`.
- `memory.factor_residency`: `gpu`, `cpu_pinned`, `cpu_staged`, `mmap_cpu`.
- `memory.output_buffers`: `fresh_allocation`, `preallocated`.
- `dtype.parameter_storage`: `fp32`, `bf16`, `fp16`, `fp8_when_supported`.
- `dtype.model_compute`: `fp32`, `bf16`, `fp16`, `fp8_when_supported`.
- `dtype.autodiff_compute`: `fp32`, `bf16`, `fp16`.
- `dtype.accumulation`: `fp32`, `bf16`, `fp16`.
- `dtype.vector`: `fp32`, `bf16`, `fp16`.
- `dtype.intermediate`: `fp32`, `bf16`, `fp16`.
- `dtype.output`: `fp32`, `bf16`, `fp16`.
- `dtype.metric_factor`: `fp32`, `bf16`, `fp16`.
- `autocast`: `off`, `cuda_fp16`, `cuda_bf16`.
- `numeric.float32_matmul_precision`: `highest`, `high`, `medium`.
- `numeric.bf16_reduced_precision_reduction`: `false`, `true`.
- `numeric.fp16_reduced_precision_reduction`: `false`, `true`.
- `numeric.deterministic_algorithms`: `false`, `true`.
- `numeric.loss_scaling`: `none`, `static_scale_with_exact_unscale`.
- `compile.enabled`: `false`, `true`.
- `compile.boundary`: `model_forward`, `transformer_block`, `attention_module`, `loss_closure`, `gradient_closure`, `jvp_closure`, `vjp_closure`, `hvp_single_vector`, `hvp_batched_vectors`, `ggn_jvp`, `ggn_loss_hessian_product`, `ggn_vjp`, `ggn_full_product`, `fisher_score_grad`, `sampled_fisher_score_grad`, `empirical_fisher_example_grad`, `metric_multiply`, `metric_inner_reduce`, `metric_sqrt_multiply`, `inverse_metric_solve`, `inverse_metric_inner_reduce`, `per_example_gradient`, `bound_operator_vector_step`, `composition_child`, `whole_operator`.
- `compile.backend`: `inductor` or a registered backend returned by the PyTorch compiler backend list that does not own CUDA graph capture.
- `compile.mode`: `None`, `default`, `max-autotune`.
- `compile.fullgraph`: `false`, `true`.
- `compile.dynamic`: `None`, `false`, `true`.
- `compile.compiled_autograd`: `false`, `true`.
- `compile.options.epilogue_fusion`: `false`, `true`.
- `compile.options.shape_padding`: `false`, `true`.
- `compile.cuda_graphs`: `false`, `true`.
- `compile.cache_state`: `cold_compile`, `warm_cache`.
- `fusion.norm`: `model_default`, `fused_rmsnorm`, `fused_layernorm`.
- `fusion.mlp`: `model_default`, `fused_mlp`.
- `fusion.rope`: `model_default`, `fused_rope`.
- `fusion.logits`: `model_default`, `fused_logits_projection`.
- `fusion.loss`: `model_default`, `fused_ce`, `fused_kl`.
- `layout.params`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`.
- `layout.vector`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`.
- `layout.output`: `parameter_tree`, `flat_contiguous`, `per_layer_flat`, `per_block_flat`, `per_shard`, `dtensor`.
- `layout.flatten_order`: `canonical_parameter_order`.
- `layout.vector_ops`: `python_loop`, `foreach`.
- `layout.contiguity`: `contiguous`, `preserve_existing_strides`.
- `layout.aliasing`: `preserve_tied_weight_aliases`.
- `layout.parametrizations`: `preserve_active_parametrizations`.
- `distributed.launch`: `single_process`, `torchrun`.
- `distributed.process_group_backend`: `nccl`, `gloo`, `ucc_when_available`.
- `distributed.local_rank_binding`: `cuda_local_rank`, `explicit_device_map`.
- `distributed.mesh_shape`: a finite positive integer tuple.
- `distributed.mesh_dim_names`: declared mesh dimension names.
- `distributed.strategy`: `single_gpu`, `fsdp2`, `hsdp`, `tensor_parallel`, `sequence_parallel`, `context_parallel`, `hybrid`.
- `dtensor.params_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.vector_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.logits_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.tangent_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.cotangent_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.output_placement`: `replicate`, `shard_dim`, `partial`.
- `dtensor.redistribute_schedule`: `none`, `before_forward`, `before_backward`, `between_operator_parts`, `before_output`.
- `fsdp.wrap_granularity`: `root`, `transformer_block`, `block_group`.
- `fsdp.reshard_after_forward`: `false`, `true`, or an admitted positive integer group size.
- `fsdp.shard_placement_fn`: `none`, `declared_fn`.
- `fsdp.mp_policy.param_dtype`: `fp32`, `bf16`, `fp16`.
- `fsdp.mp_policy.reduce_dtype`: `fp32`, `bf16`, `fp16`.
- `fsdp.mp_policy.output_dtype`: `fp32`, `bf16`, `fp16`.
- `fsdp.mp_policy.cast_forward_inputs`: `false`, `true`.
- `fsdp.offload_policy`: `none`, `cpu`.
- `fsdp.ignored_params`: a declared parameter set.
- `fsdp.dp_mesh_dims`: declared mesh dimensions.
- `tp.plan`: registered model plan id.
- `tp.qkv_projection`: `colwise`, `rowwise`, `replicated`.
- `tp.output_projection`: `rowwise`, `colwise`, `replicated`.
- `tp.mlp_up_gate`: `colwise`, `rowwise`, `replicated`.
- `tp.mlp_down`: `rowwise`, `colwise`, `replicated`.
- `tp.embedding`: `replicated`, `rowwise`, `colwise`.
- `tp.lm_head`: `replicated`, `vocab_sharded`.
- `tp.prepare_module_input`: declared input layout conversion.
- `tp.prepare_module_output`: declared output layout conversion.
- `tp.loss_parallel`: `false`, `true`.
- `sequence_parallel.enabled`: `false`, `true`.
- `sequence_parallel.norm_modules`: declared module names.
- `sequence_parallel.output_placement_policy`: `preserve_sequence_shard`, `redistribute_to_declared_output`.
- `context_parallel.enabled`: `false`, `true`.
- `context_parallel.rotate_method`: `all_gather`, `all_to_all`.
- `context_parallel.sequence_dim`: a declared sequence dimension.
- `comm.overlap`: `none`, `all_gather_overlap`, `reduce_scatter_overlap`, `both`.
- `comm.prefetch`: `none`, `forward`, `backward`, `both`.
- `comm.collective_bucket_size`: a finite positive integer domain.
These manifest rules reject contradictory rows:

- `attention.sdpa_kernel` applies only when the executable calls PyTorch SDPA.
- `attention.sdpa_kernel=priority_list` records the exact ordered backend list and sets `set_priority=True` when entering `sdpa_kernel`.
- `schedule.per_token=packed` or `attention.partition=packed_tokens` requires `input.batch_layout` to be `packed_with_inverse_permutation` or `variable_length`.
- `activation.recompute` owns the recompute mechanism. If it is not checkpoint-backed, then `checkpoint.early_stop=false`, `checkpoint.preserve_rng_state=false`, `checkpoint.determinism_check=none`, and `checkpoint.context_fn=none`.
- `activation.offload=saved_tensor_hooks_cpu` or `activation.offload=custom_saved_tensor_hooks` requires an executable saved-tensor-hooks path.
- `compile.mode="reduce-overhead"` is represented as `compile.mode=default` and `compile.cuda_graphs=true`.
- `compile.mode="max-autotune-no-cudagraphs"` is represented as `compile.mode=max-autotune` and `compile.cuda_graphs=false`.
- Backend option maps that request max autotune are represented by `compile.mode=max-autotune`.
- Backend option maps that request CUDA graph capture are represented by `compile.cuda_graphs=true`.
- Rows that set any `compile.options.*` value to `true` must use `compile.mode=None`. Rows with all compile options disabled must not set `compile.mode=None`.
- `numeric.float32_matmul_precision` owns the CUDA matmul TF32 decision.
- `tp.loss_parallel=true` requires exact cross-shard CE or KL normalization and a multi-rank agreement check.
- DTensor layout settings require matching `dtensor.*_placement` settings.
- Reduction-degrading rows require derived numeric error-bound fields.
- Sampled FisherVP rows must declare fixed sample count and fixed sample source.
- Exact categorical NLL Fisher is represented only by GGNVP with the CE or KL loss Hessian.
- Gradient materialization is derived from the AD path. `gradient.path=backward_materialized_grad` and `vjp.path=backward_materialized_grad` materialize `.grad` and read it back. All `torch.func` rows and eager `torch.autograd.grad` rows return tensor trees.
- `gradient.value_reuse=gradient_and_primal_value` requires `gradient.path=torch_func_grad_and_value` or a runtime path that explicitly returns both the primal value and gradient.
- `inverse_metric.iteration_budget` applies only to iterative solve rows.
- `attention.partition=segmented_forward_ad` requires a forward-AD operator path.
- Metric and inverse-metric rows require representation-compatible paths. `metric.multiply_path=dense_matmul` requires a dense matrix. `metric.multiply_path=factorized_multiply` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors. `metric.multiply_path=blockwise_multiply` requires block-diagonal blocks. `metric.multiply_path=streaming_multiply` requires diagonal, block-diagonal, KFAC, EKFAC, low-rank, or GGN-derived representation fields. A matrix-free metric multiplies through its declared forward operator and uses no `metric.*` path.
- `metric.block_schedule` requires block-diagonal blocks or KFAC factors. `metric.accumulation` applies only to non-dense metric multiply paths.
- Direct inverse solve paths `dense_solve`, `cholesky_solve`, `eigh_solve`, and `svd_solve` require a dense matrix; `cholesky_solve`, `eigh_solve`, and `svd_solve` over a PSD-declared metric additionally require positive damping (the shifted operator must be PD), and `eigh_solve` additionally requires a symmetric metric. `conjugate_gradient` requires an admitted metric multiply path for the same representation and a positive-definite operator (positive damping over a PSD-singular metric), and is the only solve path admitted for a matrix-free metric, inverting its forward operator iteratively. `factorized_solve` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors; EKFAC inverts in the Kronecker eigenbasis by dividing the corrected eigenvalues. `blockwise_solve` requires block-diagonal blocks. `woodbury_low_rank_solve` requires low-rank factors.
- `inverse_metric.preconditioner=block_diagonal` requires block-diagonal blocks or KFAC factors. `inverse_metric.preconditioner=factorized_metric` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors. `inverse_metric.preconditioner=matrix_free` names a sibling product whose operator is an admitted PSD curvature, and the named product must have selected rows before this row's reference check. `inverse_metric.block_schedule` requires block-diagonal blocks or KFAC factors. `inverse_metric.multi_rhs=block` applies the solve to a stacked right-hand side, and joins `metric_vp`/`inverse_metric_vp` to the vectorization applicability.
- `dtype.metric_factor` and `memory.factor_residency` are declared only by rows whose metric or inverse path uses declared or computed factors.
- Inner-product rows reuse the metric's representation compatibility. `metric_inner.reduction_path=multiply_then_reduce` requires an admitted `metric.multiply_path`; `inverse_metric_inner.reduction_path=solve_then_reduce` requires an admitted `inverse_metric.solve_path` and the same positive-damping requirement on a PSD-but-singular metric. `factored_gram` requires diagonal, KFAC, EKFAC, low-rank, or GGN-derived factors; EKFAC forms the Gram in the Kronecker eigenbasis through the corrected eigenvalues. `sqrt_apply_reduce` requires an admitted `sqrt_metric.factor_path` for the metric and composes with it. An `as_norm` row admits only `sqrt_apply_reduce`. `metric_inner.multi_rhs=block` and `inverse_metric_inner.multi_rhs=block` batch the $k$ columns into one fused reduction and join `metric_inner_vp`/`inverse_metric_inner_vp` to the vectorization applicability.
- Candidate generators do not emit representation-incompatible metric or inverse-metric rows; hand-supplied incompatible rows fail admission before reference checks.

## Execution Lowering

A candidate row becomes executable through one path:

1. Normalize aliases into manifest keys.
2. Check that every key has one owner.
3. Check that every integer axis has a finite domain or an `AutobatchDomain`.
4. Run target admission.
5. Run axis-owner admission.
6. Run adapter admission for adapter-owned keys.
7. Build a reference operation for every required check.
8. Build the measured operation.
9. Prebuild tensors, dtype conversions, input movement, layouts, and compiled callables that the row declares outside the timed region when the row owns outside-call preparation.
10. Measure only the operation body selected by the row.

Every row has a lowering owner. The owner supplies:

- Owned setting keys.
- Supported operators.
- Supported value domain.
- Admission function.
- Reference-check additions.
- Full-size-check additions.
- Function that builds the measured callable.
- Function that builds the materialized selected operator.
- Runtime fields saved on every candidate, reference row, full-size row, selected plan, and validation row.

Rows with no measured callable fail admission. Rows with a callable that ignores one of its owned settings fail the executor tests.

### Standard Runtime Lowering

The standard runtime executes PyTorch-native rows without model-library or multi-rank behavior.

Gradient:

- `gradient.path=torch_autograd_grad` lowers to `torch.autograd.grad`.
- `gradient.path=torch_func_grad` lowers to `torch.func.grad`.
- `gradient.path=torch_func_grad_and_value` lowers to `torch.func.grad_and_value`.
- `gradient.path=backward_materialized_grad` lowers to `loss.backward()` followed by reading declared `.grad` leaves.

JVP:

- `jvp.path=torch_func_jvp` lowers to `torch.func.jvp`.
- `jvp.path=forward_ad_dual` lowers to `torch.autograd.forward_ad.dual_level`, `make_dual`, and `unpack_dual`.
- `jvp.path=torch_func_linearize` lowers to `torch.func.linearize` and the returned linear function.

VJP:

- `vjp.path=torch_func_vjp` lowers to `torch.func.vjp` and the returned VJP closure.
- `vjp.path=autograd_grad_outputs` lowers to `torch.autograd.grad` with declared `grad_outputs`.
- `vjp.path=backward_materialized_grad` lowers to `output.backward(cotangent)` followed by reading declared `.grad` leaves.

HVP:

- `hvp.path=reverse_over_reverse` lowers to gradient construction followed by a second reverse-mode product.
- `hvp.path=jvp_grad` lowers to `torch.func.jvp(torch.func.grad(f))`.
- `hvp.path=autograd_functional_hvp` lowers to `torch.autograd.functional.hvp`.
- `hvp.path=autograd_functional_vhp` lowers to `torch.autograd.functional.vhp` and requires symmetry checks.
- `hvp.path=forward_ad_dual` lowers to manual forward AD over a reverse-mode gradient closure.
- `hvp.path=linearize_grad` lowers to `torch.func.linearize(torch.func.grad(f))`.

GGNVP:

- JVP rows lower through the selected JVP path.
- Loss-Hessian rows lower through exact CE/KL softmax Hessian-vector code or autodiff loss HVP over model outputs.
- `ggn.loss_hessian_kernel=dense_global` materializes the full output-space product on the declared output.
- `ggn.loss_hessian_kernel=streaming_global` streams exact global normalization without changing the loss.
- `ggn.loss_hessian_kernel=two_pass_chunked_global` computes global normalization in the first pass and chunked products in the second pass.
- VJP rows lower through the selected VJP path.

Input schedule:

- `batch.data_microbatch_size`, family-specific batch sizes, and chunk sizes split the logical batch into the declared chunks and recombine with the declared aggregation rule.
- `schedule.per_example=loop` loops over examples.
- `schedule.per_example=vmap` maps the per-example function over the leading tensor batch dimension.
- `schedule.per_example=manual_batch` builds declared subbatches and accumulates their outputs.
- `schedule.per_token=loop` loops over declared token blocks.
- `schedule.per_token=packed` consumes `input.batch_layout=packed_with_inverse_permutation` or `input.batch_layout=variable_length`, runs packed token work, and restores logical token order before output comparison.
- `schedule.gradient_accumulation=microbatch_accumulate` accumulates gradients over data microbatches in the declared accumulation dtype.
- `input.batch_layout=dense_padded` uses padded tensors and masks.
- `input.batch_layout=packed_with_inverse_permutation` builds packed tensors plus inverse permutation fields.
- `input.batch_layout=variable_length` passes declared variable-length metadata to backends that consume it.
- `input.length_grouping=exact_length_bucket` groups examples by exact length before batching and restores original order before output comparison.
- `input.host_to_device=outside_measured_call` moves inputs before timing.
- `input.host_to_device=inside_measured_call` moves inputs inside the measured callable.
- `input.residency` places inputs on CPU staged memory, CPU pinned memory, or GPU according to the row.
- `teacher_outputs` selects precomputed or recomputed teacher outputs and checks equality when recomputation is selected.

FisherVP:

- `fisher.expectation_path=explicit_full_expectation_score_rows` enumerates the declared score source exactly.
- `fisher.score_grad_path` lowers through per-score gradient construction.
- `fisher.accumulation=streaming_dot_accumulate` accumulates $g(g^\top v)$ without storing the full score matrix.
- `fisher.accumulation=materialize_score_gradients` stores the score-gradient matrix in declared layout.
- `fisher.accumulation=blockwise_score_matrix` multiplies score-gradient blocks in parameter-surface order.

Sampled FisherVP:

- `sampled_fisher.sample_source=fixed_sample_table` reads the declared sample table.
- `sampled_fisher.sample_source=fixed_seed_and_count` builds the same sample table from the declared seed and sample count before probing.
- Score-gradient and accumulation rows lower like FisherVP over the fixed sampled rows.
- Exact-Fisher comparison runs only when `sampled_fisher.exact_fisher_check=enabled_with_sampling_bound`.

EmpiricalFisherVP:

- `empirical_fisher.grad_path=torch_autograd_grad_loop` loops over examples and calls `torch.autograd.grad`.
- `empirical_fisher.grad_path=torch_func_grad` uses `torch.func.grad` per example.
- `empirical_fisher.grad_path=vmap_grad` lowers to `torch.func.vmap(torch.func.grad(...))`.
- `empirical_fisher.grad_path=backward_materialized_grad` materializes `.grad` per example and reads it back.
- Accumulation rows match FisherVP with per-example gradients.

Metric and inverse metric:

- `metric.multiply_path=dense_matmul` multiplies the declared `metric_matrix`.
- `metric.multiply_path=factorized_multiply` applies KFAC, low-rank, diagonal, or GGN-derived factors in the representation's declared order.
- `metric.multiply_path=blockwise_multiply` applies each declared metric block to the matching parameter block and writes the result back in parameter-surface order.
- `metric.multiply_path=streaming_multiply` streams declared blocks or factors from their declared residency and accumulates the output tree.
- `metric.block_schedule=layer_blocks`, `module_blocks`, or `custom_blocks` selects the block partition from the metric representation.
- `metric.accumulation=streaming` accumulates block outputs without materializing the full metric. `metric.accumulation=materialized_blocks` materializes declared blocks before multiplication.
- `inverse_metric.solve_path=dense_solve` uses `torch.linalg.solve`.
- `inverse_metric.solve_path=cholesky_solve` uses a declared or computed Cholesky factor.
- `inverse_metric.solve_path=eigh_solve` uses an eigendecomposition with declared eigenvalue handling.
- `inverse_metric.solve_path=svd_solve` uses an SVD with declared singular-value handling.
- `inverse_metric.solve_path=conjugate_gradient` runs conjugate gradient against the metric multiply operator.
- `inverse_metric.solve_path=factorized_solve` applies an inverse through declared KFAC, low-rank, diagonal, or GGN-derived factors.
- `inverse_metric.solve_path=blockwise_solve` solves each declared block and writes the result back in parameter-surface order.
- `inverse_metric.solve_path=woodbury_low_rank_solve` applies the Woodbury identity using the declared low-rank factors and diagonal base.
- `inverse_metric.preconditioner` supplies the declared preconditioner to iterative solves.
- `inverse_metric.factor_reuse=reuse_factor_across_rhs` reuses declared or computed factors across right-hand sides with identical metric representation fields.
- `inverse_metric.preconditioner=matrix_free` applies the named sibling product as the preconditioner inside the iterative solve.
- `inverse_metric.multi_rhs=block` runs the block solve (block conjugate gradient or block Lanczos) over a stacked right-hand side; `single_column` runs one column at a time.
- `sqrt_metric.factor_path` lowers to the closed-form factor square root, a Cholesky factor, an eigenbasis factor, or a matrix-free Lanczos polynomial of $f(M)v$ capped by `sqrt_metric.lanczos_iterations`, with $f(t)=t^{1/2}$ for the forward factor; the inverse-square-root operator applies the inverse factor for the closed-form kinds and a Lanczos polynomial of $(t+\lambda)^{-1/2}$ for the matrix-free kind.
- `per_example_gradient.grad_path` lowers like the empirical-Fisher gradient paths but writes the stacked per-example gradients to a leading axis instead of reducing them; `per_example_gradient.accumulation` selects a single stacked tensor or blockwise-stacked output.
- `metric_inner.reduction_path` lowers to a metric multiply followed by $U^\top(MV)$, a Kronecker-aware factor reduction into the $k\times k$ Gram, or the square-root factor adjoint applied to both blocks followed by $(L^\top U)^\top(L^\top V)$; `inverse_metric_inner.reduction_path` lowers the same three shapes over the inverse solve and the inverse-square-root factor. `metric_inner.multi_rhs=block` and `inverse_metric_inner.multi_rhs=block` run the reduction over the stacked block; `single_column` runs one column at a time.
- Damped inverse rows check $\|(M+\lambda I)x-v\|/\|v\|$.
- `inverse_metric.iteration_budget` caps iterative solves only; direct solve rows reject it.

Numeric backend:

- `dtype.parameter_storage` stores model parameters in the declared dtype before probing.
- `dtype.model_compute` casts model compute inputs and module parameters at the declared boundary.
- `dtype.autodiff_compute` casts tensors entering AD closures.
- `dtype.accumulation` controls reduction accumulation dtype.
- `dtype.vector`, `dtype.intermediate`, `dtype.output`, and `dtype.metric_factor` cast the corresponding tensors at declared boundaries.
- `autocast` enters the declared PyTorch autocast context.
- `numeric.float32_matmul_precision` sets PyTorch float32 matmul precision before probing.
- Reduced-precision reduction flags set the corresponding PyTorch backend flags before probing.
- `numeric.deterministic_algorithms` sets PyTorch deterministic algorithms before probing.
- `numeric.loss_scaling=static_scale_with_exact_unscale` scales the declared scalar or score-gradient path and applies the exact unscale law before returning the vector product.

Composition:

- `composition.execution=materialize_each_child` calls selected child materializers and executes them in declared order.
- `composition.execution=stream_child_outputs` passes each child output directly into the next child.
- `composition.execution=fuse_adjacent_children` builds a fused callable for adjacent compatible children and validates every child output.
- `composition.execution=compile_whole_composition` compiles the composed callable when compile settings admit it.
- `composition.child_evaluation=selected_child_rows` uses already selected child-family rows named in the ordered `children` list.
- `composition.child_evaluation=inline_child_lowering` builds child lowerings inside the parent row using the ordered `children` list.
- `composition.validation=validate_each_child` runs each child reference check at the child boundary.
- `composition.validation=validate_composed_output` runs the full composed-output reference check after the last child.

### Attention Execution Lowering

Attention execution is model-library agnostic. A model adapter supplies an attention-location descriptor that identifies Q, K, V, mask, position, dropout, scale, RoPE, softcap, cache, and output locations. The attention executor owns PyTorch SDPA contexts and package exact attention kernels.

Core attention frontend lowering:

- `pytorch_sdpa_direct` calls `torch.nn.functional.scaled_dot_product_attention` at the declared attention location.
- `patched_eager` installs a package-owned differentiable eager attention wrapper at the declared attention location.
- `packed_exact` executes exact packed-token attention and restores inverse token order.
- `blockwise_exact` executes exact blockwise query attention and preserves masks, causality, RoPE, softcaps, and logit scaling.

Attention partition and padding lowering:

- `attention.partition=full` runs the full declared attention problem in one logical attention call.
- `attention.partition=packed_tokens` consumes packed or variable-length input layout and runs attention over packed tokens.
- `attention.partition=blockwise_queries` splits queries into declared blocks, runs exact attention for each block against the full key/value set, and concatenates outputs in query order.
- `attention.partition=segmented_forward_ad` runs forward-AD tangents over declared attention segments and combines exact tangents in logical order.
- `attention.padding=dense_padded` keeps padded attention tensors and applies the declared mask.
- `attention.padding=unpadded_packed` removes padding before the attention backend and restores padded logical output shape after the backend.

SDPA kernel lowering:

- `attention.sdpa_kernel=math` enters `sdpa_kernel(SDPBackend.MATH)`.
- `attention.sdpa_kernel=flash_attention` enters `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`.
- `attention.sdpa_kernel=efficient_attention` enters `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`.
- `attention.sdpa_kernel=cudnn_attention` enters `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`.
- `attention.sdpa_kernel=overrideable` enters `sdpa_kernel(SDPBackend.OVERRIDEABLE)`.
- `attention.sdpa_kernel=priority_list` enters `sdpa_kernel([...] , set_priority=True)` with the declared ordered `SDPBackend` list.

Full-size attention checks:

- Non-math SDPA kernels require full-size agreement checks.
- FlashAttention, FlexAttention, paged attention, registered attention, packed attention, blockwise attention, CUDA graph capture, max-autotune compilation, fused kernels, tensor-parallel loss, and context-parallel attention require full-size agreement checks at a size that triggers the selected backend.
- Tiny references still run, but they do not replace the full-size gate.

### Transformers Lowering

Transformers rows are executable through `vptune.adapters.transformers`.

Model loading:

- `load_transformers_model` calls a Transformers-style `from_pretrained` loader with explicit model id, revision, dtype, selected attention frontend, selected custom attention id when present, and `use_cache`.
- The adapter resolves the Transformers-owned `attention.frontend` values to the exact `attn_implementation` string sent to Transformers.
- `model.set_attn_implementation(...)` is the runtime switch when a row changes only the Transformers attention frontend on an already loaded compatible model.

Transformers frontend lowering:

- `transformers_eager` selects `attn_implementation="eager"`.
- `transformers_sdpa` selects `attn_implementation="sdpa"`.
- `transformers_flash_attention_2`, `transformers_flash_attention_3`, and `transformers_flash_attention_4` select the matching Transformers backend string.
- `transformers_flex_attention` selects `attn_implementation="flex_attention"`.
- `paged|eager`, `paged|sdpa`, `paged|flash_attention_2`, `paged|flash_attention_3`, and `paged|flash_attention_4` select the exact paged backend string.
- `registered_transformers_attention` registers the declared attention function through `AttentionInterface` and registers the declared mask function through `AttentionMaskInterface` before probing.

### Compile Lowering

Compile rows build the callable named by `compile.boundary`.

- `compile.enabled=false` runs the eager callable.
- `compile.enabled=true` wraps the callable with `torch.compile`.
- `compile.backend` is passed to `torch.compile`.
- `compile.mode` is passed as `None`, `default`, or `max-autotune`.
- `compile.fullgraph` and `compile.dynamic` are passed to `torch.compile`.
- `compile.compiled_autograd=true` enables the PyTorch compiled-autograd setting around the compiled callable.
- `compile.options.epilogue_fusion` and `compile.options.shape_padding` are passed as backend options.
- `compile.cuda_graphs=true` passes the backend option that requests CUDA graph capture for the selected backend.
- `compile.cache_state=cold_compile` clears the package-managed compile state before timing.
- `compile.cache_state=warm_cache` compiles before steady-state timing.

Compile rows record compile time, first-call time, steady-state samples, recompile count, graph-break status, compiled-autograd status, CUDA graph capture status, and memory during compile and steady-state calls.

### Fusion Lowering

Fusion rows replace a declared model subpath with the selected fused implementation before reference and timing:

- `fusion.norm=fused_rmsnorm` replaces declared RMSNorm modules with the registered fused RMSNorm implementation.
- `fusion.norm=fused_layernorm` replaces declared LayerNorm modules with the registered fused LayerNorm implementation.
- `fusion.mlp=fused_mlp` replaces declared MLP modules with the registered fused MLP implementation.
- `fusion.rope=fused_rope` replaces declared RoPE application with the registered fused RoPE implementation.
- `fusion.logits=fused_logits_projection` replaces the declared output projection with the registered fused logits projection.
- `fusion.loss=fused_ce` replaces CE loss computation with the registered fused CE implementation and requires exact global normalization.
- `fusion.loss=fused_kl` replaces KL loss computation with the registered fused KL implementation and requires exact global normalization.

Every fused implementation must declare supported AD order. HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, and compositions containing them require a higher-order agreement check at a backend-triggering size.

### Checkpoint And Memory Lowering

Checkpoint-backed recompute rows call `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`.

- `activation.recompute=checkpoint_non_reentrant_by_layer` and `activation.recompute=checkpoint_selective` lower through checkpoint.
- `activation.recompute=manual_recompute` lowers through package-owned recompute closures.
- `memory.primal_outputs=recompute` drops primal outputs after the primal call and rebuilds them before reuse.
- `memory.jvp_outputs=recompute` drops JVP outputs after their first use and rebuilds them before reuse.
- `memory.output_cotangents=recompute` rebuilds output cotangents instead of retaining them.
- `activation.offload=saved_tensor_hooks_cpu` lowers through PyTorch saved-tensor hooks that move saved tensors to CPU.
- `activation.offload=custom_saved_tensor_hooks` lowers through the declared hook pair.
- `memory.vector_residency`, `memory.intermediate_residency`, and `memory.factor_residency` execute explicit device movement before or inside the measured call according to their owner.
- `memory.output_buffers=preallocated` creates buffers before timing and writes into them during timing.

Rows that recompute RNG-consuming regions must preserve RNG state and pass deterministic recomputation checks.

### Layout Lowering

Layout rows own tensor representation and reconstruction:

- `parameter_tree` uses the parameter-surface tree.
- `flat_contiguous` flattens leaves in canonical parameter order.
- `per_layer_flat` and `per_block_flat` flatten leaves by declared layer or block groups.
- `per_shard` uses rank-local layout under the distributed adapter.
- `dtensor` uses DTensor placements declared by distributed settings.
- `layout.vector_ops=foreach` uses PyTorch foreach operations for elementwise vector operations when dtype and device groups admit them.
- `layout.contiguity=contiguous` materializes contiguous tensors before the measured call when the row owns outside-call preparation.
- `layout.aliasing=preserve_tied_weight_aliases` reconstructs tied leaves through shared storage groups.
- `layout.parametrizations=preserve_active_parametrizations` uses active parametrization views.

All matrix-vector paths flatten vectors in canonical parameter order unless the operator spec declares an external dense matrix convention. FisherVP, sampled FisherVP, empirical FisherVP, GGNVP, HVP, and VHP reconstruct outputs against the parameter surface, not against vector input order.

### Distributed Lowering

Distributed rows execute through `vptune.adapters.distributed`.

- `distributed.launch=torchrun` starts one process per declared rank and initializes the process group.
- `distributed.process_group_backend` selects `nccl`, `gloo`, or `ucc_when_available`.
- `distributed.local_rank_binding` maps ranks to devices.
- `distributed.mesh_shape` and `distributed.mesh_dim_names` build the `DeviceMesh`.
- `dtensor.*_placement` creates `Replicate`, `Shard(dim)`, or `Partial` placements.
- `dtensor.redistribute_schedule` calls `redistribute` at the declared operator boundary.
- `distributed.strategy=fsdp2` calls `torch.distributed.fsdp.fully_shard`.
- `distributed.strategy=hsdp` uses FSDP2 with declared sharded and replicated mesh dimensions.
- `distributed.strategy=tensor_parallel` calls `torch.distributed.tensor.parallel.parallelize_module` with the declared plan.
- `distributed.strategy=sequence_parallel` applies declared sequence-parallel module styles.
- `distributed.strategy=context_parallel` uses PyTorch context-parallel APIs with `all_gather` or `all_to_all`.
- `comm.overlap`, `comm.prefetch`, and `comm.collective_bucket_size` configure collective scheduling when the selected distributed API exposes the setting.
- `fsdp.wrap_granularity` selects the module set passed to `fully_shard`.
- `fsdp.reshard_after_forward` is passed to FSDP2 reshard control.
- `fsdp.shard_placement_fn` is passed as the declared shard placement function.
- `fsdp.mp_policy.*` builds the FSDP2 mixed-precision policy.
- `fsdp.offload_policy` builds the FSDP2 offload policy.
- `fsdp.ignored_params` is passed to FSDP2 as the excluded parameter set.
- `fsdp.dp_mesh_dims` selects the data-parallel mesh dimensions for HSDP rows.
- `tp.qkv_projection`, `tp.output_projection`, `tp.mlp_up_gate`, `tp.mlp_down`, `tp.embedding`, and `tp.lm_head` map declared modules to rowwise, colwise, replicated, or vocab-sharded tensor-parallel styles.
- `tp.prepare_module_input` and `tp.prepare_module_output` install declared layout conversion hooks.
- `tp.loss_parallel=true` runs the sharded CE or KL loss with exact global normalization.
- `sequence_parallel.norm_modules` applies sequence-parallel placement to the declared normalization modules.
- `sequence_parallel.output_placement_policy` controls whether output remains sequence-sharded or is redistributed.
- `context_parallel.rotate_method` selects `all_gather` or `all_to_all`.
- `context_parallel.sequence_dim` selects the tensor dimension split by context parallelism.

Distributed rows run reference checks on the same logical batch as single-device anchors. Multi-rank full-size checks compare the logical output after gathering or declared output placement conversion.

## Search Algorithm

Search runs over a family DAG.

Search also runs over the axis groups from the feature manifest:

- Class A and Class B are search annotations on keys that still belong to one Class C primary group.
- Class C primary groups form the complete axis partition.
- Merge rules join Class C groups before candidate generation for coupled settings such as packed attention, attention-module compilation, operator-part compilation, fused kernels, DTensor placement, mixed-precision distributed reduction, and factorized inverse solves.
- `admission` builds the candidate table and runs no full-size timings.
- `smoke` measures one baseline row per operator family and one representative row from each merged Class C group.
- `fast` runs a coarse joint search over operator path, vectorization, dtype, attention, and eager execution, then compiles top eager rows.
- `balanced` searches each merged Class C group with successive halving, keeps the configured top count, crosses the retained group winners, compiles top eager rows at multiple boundaries, and runs selected-plan validation.
- `thorough` expands top coupled groups with more dtype, compile, layout, and chunking variants, evaluates distributed rows on all declared ranks, and evaluates compile amortization across declared call horizons.
- `exhaustive` evaluates the full admitted cross product and is allowed only for explicitly bounded candidate sets.

Class A keys are swept inside a fixed operator path, dtype, attention, layout, and distributed setting:

- `layout.vector_ops`
- `gradient.value_reuse`

Class B keys are swept only after the row fixes operator path, attention frontend, compile state, and distributed strategy:

- `input.residency`
- `input.host_to_device`
- `compile.cache_state`
- `teacher_outputs`

Class C primary groups form a partition:

- `ad_lowering`: `gradient.*`, `jvp.*`, `vjp.*`, `hvp.*`, `ggn.*`, `fisher.*`, `sampled_fisher.*`, `empirical_fisher.*`, `per_example_gradient.*`, `composition.*`, `vectorization.*`, and `call.*`.
- `attention_dispatch`: `attention.frontend`, `attention.sdpa_kernel`, `attention.custom_kernel_id`, `attention.mask_formatter_id`, `attention.partition`, and `attention.padding`.
- `input_schedule`: `batch.*`, `chunk.*`, `schedule.*`, `input.*`, and `teacher_outputs`.
- `activation_memory`: `checkpoint.*`, `activation.*`, `memory.primal_outputs`, `memory.jvp_outputs`, `memory.output_cotangents`, `memory.vector_residency`, and `memory.intermediate_residency`.
- `numeric_backend`: `dtype.*`, `autocast`, and `numeric.*`.
- `distributed_layout`: `layout.*`, `dtensor.*`, `distributed.*`, `fsdp.*`, `tp.*`, `sequence_parallel.*`, `context_parallel.*`, and `comm.*`.
- `compile`: `compile.*` and `memory.output_buffers`.
- `fusion`: `fusion.*`.
- `metric_storage`: `metric.*`, `metric_inner.*`, `sqrt_metric.*`, and `memory.factor_residency`.
- `inverse_solve`: `inverse_metric.*` and `inverse_metric_inner.*`.

Class C merge rules:

- `attention.partition=packed_tokens` merges `attention_dispatch` with `input_schedule`.
- `compile.boundary=attention_module` merges `compile` with `attention_dispatch`.
- A compile boundary that names an operator part merges `compile` with `ad_lowering`.
- Any `fusion.*` value other than `model_default` merges `fusion` with `ad_lowering`.
- DTensor placement for params, vectors, logits, tangents, cotangents, or outputs merges `distributed_layout` with `ad_lowering`.
- `fsdp.mp_policy.reduce_dtype=bf16` or `fsdp.mp_policy.reduce_dtype=fp16` merges `distributed_layout` with `numeric_backend`.
- A factorized metric or factorized preconditioner in an inverse row merges `inverse_solve` with `metric_storage`.
- An `inverse_metric_inner.reduction_path` of `factored_gram` or `sqrt_apply_reduce` merges `inverse_solve` with `metric_storage`, since the reduction reads the same factors.

After merging, the search strategy must search each merged group jointly or use a staged method that carries top rows forward before crossing groups.

1. Build candidate rows for every family and cohort assignment.
2. Validate family names, row ids, changed axes, target allowances, admissions, and settings.
3. Validate the family DAG has no cycles.
4. Run admission checks for every row.
5. Run reference checks for admitted rows.
6. Write `ReferenceFailed` full-size rows for rows whose references fail.
7. Tune DAG roots per cohort assignment.
8. Select the best row for each completed root family inside the same cohort assignment.
9. Tune downstream families only after selected dependency rows are available inside the same cohort assignment.
10. Write `PrerequisiteFailed` rows for rows blocked by failed dependencies.
11. Compare complete cohort assignments.
12. Write selected settings.
13. Run selected-plan validation over selected rows in dependency order when a validation problem is declared.

Failure propagation follows the DAG. A failed family blocks only its descendants.

Within a family, `vptune` measures every explicit candidate row or delegates one integer candidate axis to Autobatch when the candidate generator declares an `AutobatchDomain`. Reference checks run before Autobatch. `autobatch.find` receives only reference-passed values and probes the package operation directly. The selected Autobatch value is written into the selected full-size row as selection metadata. Replay uses that row for Autobatch families when it is current, stable, reference-passed, and passed.

Composition reference checks may declare typed child descriptors. Each child descriptor has a name, candidate, candidate component, anchor component, reference check, and input identity. During parent reference checking, the child reference check runs on the current component input. Passed child rows are saved before the parent composition row, and the parent row records ordered child reference descriptors: child name, family, row id, check name, input identity, candidate settings, thresholds, dependency identities, cohort assignment identity, generator identity, and child status.

A cohort assignment is a general constraint over settings, not a dtype-only rule. Dtype coherence is represented as `CohortConstraint(name="dtype_model_compute", settings_keys=("dtype.model_compute",), ...)`. Search state is `(family, cohort_assignment, dependency_selection)`.

Autobatch domains write ordinary candidate, reference, full-size, and failed rows for observed values. The domain fields include the ordered values, value-to-settings function id, admission fields, timing policy, target devices, and reuse fields.

Every probe row records:

- Package version.
- PyTorch version.
- Model adapter id.
- Operator spec fields.
- Candidate settings.
- Dependency identity records.
- Input data signature.
- Vector signature.
- Target device signature.
- Materializer identity.
- Adapter identity.

Current-record validation:

- Reference rows are current only when record type, status, input signature, candidate settings, thresholds, family, row id, check name, dependency identities, cohort assignment identity, package version, and schema version match the current run by direct field equality.
- Candidate rows are current only when record type, status, input signature, settings, changed axes, family, row id, generator id, generator version, admission status, admission error, migration source id, dependency identity records, selected dependency identities, cohort assignment identity, operator spec identity, package version, and schema version match the current run by direct field equality.
- The operator spec identity bucket holds the typed-object fields that are fixed problem fields rather than sweep axes: the loss/output/likelihood closed-set fields and wrapped-callable value versions, the metric kind and value version (and the wrapped product name for a `matrix_free` metric), the sample-source seed, count, and table identity, the damping kind and values, the solver residual tolerance `tol`, and the inner-product norm declaration `as_norm`. A different `from_scalar` value version, sample seed, damping value, `tol`, or `as_norm` setting is a different row.
- Composition rows additionally compare the composition expression fields by direct field equality: the `Combine` tree shape, every coefficient and constant, every child-name and `source` leaf, and the implied `call_inputs`. A different `linear_combination` coefficient is a different row.
- Full-size rows are current only when record type, status, input signature, settings, family, row id, dependency identities, selected dependency identities, cohort assignment identity, operator spec identity, generator identity, package version, and schema version match the current run by direct field equality.
- A failed row with `ReferenceFailed` or `NoPassedCandidate` is non-terminal. It must be rechecked when reference state or dependency selection changes.
- Runtime failures and OOM failures are terminal for that row and input signature. If the operation enters the timed region, the failed row keeps elapsed and memory samples from that call.
- Selected-plan validation summaries are current only when the saved plan fields, validation order, validation row descriptors, validation row statuses, and summary fields match the current run and loaded validation rows by direct field equality.
- Replay rejects a supplied selected-plan validation summary unless the summary and every validation row passed.
- Replay requires a `ReplayContext`. It checks saved rows against the current model, data, vectors, target, runtime, adapter, materializer, family input signatures, and selection policy.
- Replay requires saved candidate rows as inputs alongside saved reference and full-size rows. Candidate rows must pass direct field checks for candidate spec fields, dependency identities, generator identity, input signature, and schema-valid row content.
- Replay recomputes selection in dependency order. For downstream families, replay filters rows to those whose selected dependency identities match rows already selected inside the same cohort assignment. A saved summary is accepted only when recomputed selected rows and the recomputed selected cohort assignment match the saved plan.

## Measurement Protocol

Each full-size candidate run does:

1. Empty unused allocator cache.
2. Run the declared pre-call reset hook.
3. Reset peak memory stats on every measured device.
4. Synchronize every measured device.
5. Start the wall-clock timer.
6. Execute the operation.
7. Synchronize every measured device.
8. Record elapsed time.
9. Record memory sample.
10. Empty unused allocator cache.

The first call is a probe call. Timing plan:

- First elapsed below 60 seconds: two warmups and five measured calls.
- First elapsed from 60 seconds to below 600 seconds: one warmup and three measured calls.
- First elapsed at least 600 seconds: one measured call, the probe call.

Compiled rows measure compile work separately from steady-state work. For `compile.cache_state=cold_compile`, the compile call records compile time and memory, then the timing-plan bucket is chosen from the first post-compile steady-state call. For `compile.cache_state=warm_cache`, the callable is compiled before the probe call and the bucket is chosen from the first steady-state call.

These thresholds are package defaults. A caller can declare a different timing policy, and the policy identity is part of the saved record.

Memory stability:

- One memory sample is stable.
- Multiple samples are stable when no later `post_allocated_mib` and no later `post_reserved_mib` exceeds the first measured sample.
- Distributed and model-parallel rows apply this rule per rank and per device, then reduce status across ranks.

OOM and runtime errors:

- CUDA OOM records a failed row with error type and message.
- Runtime errors record a failed row with error type and message.
- Failed rows remain visible to selection and reporting.

Every memory sample contains:

- `peak_allocated_mib`
- `peak_reserved_mib`
- `post_allocated_mib`
- `post_reserved_mib`
- `device_memory_used_mib` when CUDA exposes it
- `rank`
- `device`

Measurement lifecycle:

- Candidate code runs on the current stream unless the target declares a stream policy.
- Multi-device rows synchronize all participating devices before and after the operation.
- Distributed rows use a barrier before the timed region and after the timed region.
- Output tensors are reduced to an output signature after the measured call; retained tensors must be declared in the row output.
- Exception timing is recorded only when the operation enters the timed region.
- `empty_cache` is part of the measurement protocol and therefore part of the memory backend identity in the tuning input signature.

## Selection Rule

Inside one family:

1. Accept rows with passed references, current input signature, current replay identity fields, passed full-size status, stable memory, and all required full-size or kernel-triggering agreement checks.
2. Compute the selection score per row. The timing source is eager single-rank, eager distributed, compiled single-rank, or compiled distributed. Single-rank rows use local elapsed samples. Distributed rows use the maximum rank elapsed sample after barriers. Eager rows use median steady-state elapsed seconds. Compiled rows use $((1+R)T_{\mathrm{compile}}/N)+T_{\mathrm{steady}}$, where $R$ is the measured recompile count and $N$ is the declared call horizon. Compiled distributed rows use the same formula with rank-maximum compile time and rank-maximum steady-state time. Memory tie breaking uses the row's selected memory reduction.
3. Find the fastest selection score.
4. Keep rows with selection score at most `fastest * 1.05`.
5. Select the row with smallest peak reserved memory among those rows.

Across cohort assignments:

1. Build one complete selected row set per assignment.
2. Sum the selected row scores from family selection. Compiled rows keep compile amortization, distributed rows keep rank-maximum timing, and compiled distributed rows keep both.
3. Find the fastest total.
4. Keep cohorts with total elapsed at most `fastest * 1.05`.
5. Select the cohort with smallest summed selected memory score.

A cohort is complete only when every family required by the constraint has an accepted row for that assignment and every downstream row's dependency identity names the upstream row selected inside the same assignment.
Accepted cohort rows are checked against the current family input signature, not their own saved input signature.

The selected settings are deterministic for fixed saved rows.

Selection policy fields:

- `speed_statistic`: `median_elapsed_seconds`.
- `compiled_speed_statistic`: compile-amortized steady-state score.
- `distributed_speed_statistic`: scalarized global wall time.
- `rank_memory_reduction`: `max_peak_allocated`, `max_peak_reserved`, or `sum_peak_reserved`.
- `near_fastest_multiplier`: `1.05`.
- `tie_breaker`: `min_peak_reserved_mib`.
- `cohort_speed_statistic`: sum of selected row scores.
- `cohort_tie_breaker`: sum of selected row memory scores.
- `accepted_status`: passed references, current input signature, passed full-size row, required full-size agreement rows, and stable memory.

The policy is recorded in every selection summary.

## PyTorch Admission Rules

`functional_call` rows must declare:

- Parameter dictionary keys.
- Buffer dictionary keys.
- `tie_weights` value.
- `strict` value.
- Active parametrization key policy. Disabled parametrization rows fail admission until a package-owned execution path implements that behavior.
- Whether the called module mutates parameters or buffers in place.
- Declared mutated parameter and buffer keys when mutation is enabled.
- Module train/eval mode.

Rows using `torch.func` transforms must pass admission checks for:

- No nested `torch.autograd.grad` or `backward` inside transformed pure functions.
- No unsupported `out=` operations.
- No data-dependent Python control flow over batched tensors.
- No `.item()` calls on transformed tensors.
- No dynamic-shape outputs under `vmap`.
- Explicit `vmap` randomness mode: `error`, `same`, or `different`.
- Forward AD coverage for `jvp`, `jacfwd`, and `hessian` rows.

The direct standard runtime requires torch.func admission fields for every torch.func-transform row:

- `gradient.path=torch_func_grad`
- `gradient.path=torch_func_grad_and_value`
- `jvp.path=torch_func_jvp`
- `jvp.path=torch_func_linearize`
- `vjp.path=torch_func_vjp`
- `hvp.path=jvp_grad`
- `hvp.path=linearize_grad`
- `ggn.jvp_path=torch_func_jvp`
- `ggn.jvp_path=torch_func_linearize`
- `ggn.vjp_path=torch_func_vjp`
- `fisher.score_grad_path=torch_func_grad`
- `fisher.score_grad_path=vmap_grad`
- `sampled_fisher.score_grad_path=torch_func_grad`
- `sampled_fisher.score_grad_path=vmap_grad`
- `empirical_fisher.grad_path=torch_func_grad`
- `empirical_fisher.grad_path=vmap_grad`

Forward-mode rows require forward-AD admission fields:

- `jvp.path=torch_func_jvp`
- `jvp.path=forward_ad_dual`
- `hvp.path=jvp_grad`
- `hvp.path=forward_ad_dual`
- `ggn.jvp_path=torch_func_jvp`
- `ggn.jvp_path=forward_ad_dual`

Rows in the forward-mode list must declare `requires_forward_ad=True` and `forward_ad_supported=True`. Manual forward-AD rows use `forward_ad_dual` and do not require unrelated torch.func transform-limit fields.

Rows that provide any `functional_call` admission field must provide all `functional_call` admission fields. The direct standard runtime applies the same validation as the registry.

Checkpoint-backed recompute rows must record:

- `use_reentrant=false`.
- `preserve_rng_state`.
- `determinism_check`.
- `context_fn`.
- `early_stop`.
- Whether the checkpointed function moves tensors to a device not present in its inputs.
- Whether the checkpointed function depends on global mutable state.

HVP rows may use `vhp` as an execution candidate only when the operator passes HVP symmetry and finite-difference checks on the same reference input. The selected row still reports an HVP result.

Compile rows must record:

- Callable boundary.
- Backend.
- Mode.
- Fullgraph flag.
- Dynamic-shape flag.
- Compiled-autograd flag.
- Backend options used by package-owned compile rows.
- CUDA graph capture flag.
- Compile cache state.
- Graph-break status.
- Recompile count.

Compile admission rules:

- `compile.enabled=false` forbids other compile fields except fields that take their disabled value.
- `compile.enabled=true` requires an executable compile boundary.
- `compile.fullgraph=true` rejects any graph break.
- `compile.compiled_autograd=true` requires a backward or higher-order AD path whose backward graph is inside the selected compile boundary.
- `compile.cuda_graphs=true` requires static shapes at the selected boundary unless the backend explicitly admits the shape pattern.
- `compile.mode=max-autotune` and backend option maps that request max autotune are one decision; the normalized row uses `compile.mode=max-autotune`.
- `compile.mode=default` with `compile.cuda_graphs=true` represents PyTorch reduce-overhead behavior.
- `compile.mode=max-autotune` with `compile.cuda_graphs=false` represents PyTorch max-autotune without CUDA graph capture.
- `compile.options.*=true` rows must set `compile.mode=None`.
- Rows with all `compile.options.*=false` must not set `compile.mode=None`.

SDPA rows must record:

- Attention frontend.
- Whether the executable calls PyTorch SDPA.
- SDPA backend or ordered backend list.
- Whether priority order is active.
- Dropout probability.
- Mask semantics.
- Causal policy.
- GQA policy.
- Effective runtime dtype.

SDPA admission rules:

- `attention.sdpa_kernel` is valid only for rows whose executable calls PyTorch SDPA.
- `attention.sdpa_kernel=priority_list` requires a nonempty ordered backend list.
- FlashAttention rows require effective runtime dtype `float16` or `bfloat16`.
- GQA rows require query heads divisible by key/value heads and key heads equal to value heads.
- Eval references require `dropout_p=0.0` unless the randomness policy declares stochastic attention.
- Rows that request unavailable backends fail admission with the backend failure reason.

Full-size agreement gates are mandatory for:

- Non-math SDPA kernels.
- Shape-dependent attention backends.
- FlashAttention, FlexAttention, paged attention, registered attention, packed attention, and blockwise attention.
- CUDA graph capture.
- Max-autotune compilation.
- Fused kernels.
- Packed kernels.
- Sharded reductions.
- Tensor-parallel loss.
- Context-parallel attention.

The full-size gate runs at an input size that triggers the selected backend. Selection ignores rows whose tiny references pass but whose full-size gate is missing.

## Attention Executor

The attention executor is part of core `vptune`, not a Transformers adapter dependency. It owns:

- `pytorch_sdpa_direct`
- `patched_eager`
- `packed_exact`
- `blockwise_exact`
- `attention.sdpa_kernel`
- `attention.partition`
- `attention.padding`

Every model adapter that wants package-owned attention execution must provide an attention-location descriptor. The descriptor names the tensors, masks, positions, cache fields, layout, and model-specific semantic fields that the core attention executor needs.

## Pilot Adapter

The pilot adapter is a package-edge adapter. It lowers caller-declared families and problems into a `TuningRun`, checks readiness for required selected families and their selected dependencies, and converts a passed plan into selected settings for the caller.

Readiness expands each requested family through its selected dependency identities. A required family is ready only when the selected family and every selected dependency family have passed full-size rows, passed references, and current replay identity fields.

Selected-settings conversion checks selected-plan validation rows when the plan declares validation. Validation rows must appear in the plan validation order and must match the selected candidate id, candidate settings, dependency identities, cohort assignment, generator id, generator version, validation status, validation replay identity fields, and selected-plan validation input signature.

## Transformer Adapter

The Transformers model adapter records an explicit model identity:

- Transformers package version.
- Model config fields.
- Module identity, including tied-weight groups, buffers, training mode, devices, dtypes, and active parametrizations.
- Source revision when available.
- Dtype policy.
- Tokenizer identity.
- Adapter rule identity.

`vptune.adapters.load_transformers_model(...)` calls a Transformers-style `from_pretrained` loader with explicit `model_name_or_path`, `revision`, `torch_dtype`, `attention_frontend`, registered-kernel id when the selected frontend needs one, and `use_cache`. It resolves the selected frontend to the loader's `attn_implementation` value before probing.

Attention implementation is part of candidate settings.

Supported rows:

- `transformers_eager`
- `transformers_sdpa`
- `transformers_flash_attention_2`
- `transformers_flash_attention_3`
- `transformers_flash_attention_4`
- `transformers_flex_attention`
- `paged|eager`
- `paged|sdpa`
- `paged|flash_attention_2`
- `paged|flash_attention_3`
- `paged|flash_attention_4`
- `registered_transformers_attention`

PyTorch SDPA kernel rows are `math`, `flash_attention`, `efficient_attention`, `cudnn_attention`, `overrideable`, and `priority_list`. `priority_list` records the exact backend order. Auto selection is represented only by this priority list.

Admission checks are explicit:

- FlashAttention rows require effective runtime dtype `float16` or `bfloat16`; `dtype.model_compute` overrides `dtype.parameter_storage`.
- Unknown Transformers attention implementations fail direct admission and axis admission.
- SDPA rows record unsupported settings such as `output_attentions=True`.
- Model-specific softcap is included in the adapter signature.
- Rows that cannot preserve the model's mathematical attention semantics fail admission.
- `module_mode` is recorded for every attention row.
- `dropout_p` is recorded and must be `0.0` for eval references unless the randomness policy declares stochastic attention. Missing `dropout_p` fails eval admission.
- Boolean mask semantics are recorded for each attention path.
- Causal mask handling is recorded.
- GQA is admitted only when query heads divide key/value heads and key heads equal value heads.
- Forced fused-kernel rows record the exact failure reason when the kernel is unavailable.
- Backend numerical behavior is part of the row signature.
- Determinism flags are part of the target identity.
- Padding limits for FlashAttention rows are part of admission.
- Packed target-row attention rows declare a packed attention id, semantic identity, target-row axis, positive target-row count, explicit softcap preservation, explicit mask preservation, mask semantics, and causal policy.
- Blockwise second-derivative attention rows declare a blockwise attention id, semantic identity, positive block size, softcap-preservation flag, and mask-preservation flag. Both preservation flags must be `true`.
- Patched attention reference checks compare outputs and VJPs on explicitly declared differentiable arguments.
- Non-math attention backends and shape-dependent attention backends require full-size agreement checks for every operator family whose model forward executes attention.

Attention rows never change semantics implicitly.

Transformers executor API:

```python
def load_transformers_model(
    model_cls: TransformersModelLoader,
    *,
    model_name_or_path: str,
    revision: str,
    torch_dtype: torch.dtype,
    attention_frontend: str,
    attention_custom_kernel_id: str | None,
    use_cache: bool,
) -> torch.nn.Module: ...

def transformers_attn_implementation(
    attention_frontend: str,
    attention_custom_kernel_id: str | None,
) -> str: ...

def transformers_attention_axis() -> AxisDescriptor: ...

def check_patched_attention_output_reference(...) -> ReferenceResult: ...
def check_patched_attention_vjp_reference(...) -> ReferenceResult: ...
```

The adapter must provide operation factories for:

- Loading a model with the selected attention frontend.
- Switching an already loaded compatible model through `set_attn_implementation`.
- Entering PyTorch `sdpa_kernel` for SDPA rows.
- Registering custom attention and mask functions before probing.
- Installing patched eager attention wrappers.
- Running exact packed-token attention.
- Running exact blockwise attention.

The adapter must provide reference and full-size checks for every listed attention frontend. FlashAttention-2, FlashAttention-3, FlashAttention-4, FlexAttention, paged attention, registered attention, packed attention, and blockwise attention must prove output equality and the derivative equality required by the selected operator family.

## Distributed Adapter

Distributed rows are selected by the same measurement and stability rules as single-device rows.

The distributed adapter must record:

- Device mesh.
- Rank count.
- Per-rank placement.
- Global parameter surface.
- Rank-local memory samples.
- Global maxima for peak allocated, peak reserved, post allocated, and post reserved memory.
- Rank-local status.
- Global status.
- Communication settings.
- Selected sharding settings.

The supported distributed axes are:

- `distributed.launch`
- `distributed.process_group_backend`
- `distributed.local_rank_binding`
- `distributed.mesh_shape`
- `distributed.mesh_dim_names`
- `distributed.strategy`
- `dtensor.*_placement`
- `dtensor.redistribute_schedule`
- `fsdp.wrap_granularity`
- `fsdp.reshard_after_forward`
- `fsdp.shard_placement_fn`
- `fsdp.mp_policy.*`
- `fsdp.offload_policy`
- `fsdp.ignored_params`
- `fsdp.dp_mesh_dims`
- `tp.*`
- `sequence_parallel.*`
- `context_parallel.enabled`
- `context_parallel.rotate_method`
- `context_parallel.sequence_dim`
- `comm.*`

Every rank must agree on selected settings before materialization.

Distributed records require adapter id, adapter version, device mesh, placements, and communication identity. Rank-local status rows, memory samples, and selected-setting rows must report the same rank set, and that set must equal `{0, ..., expected_rank_count - 1}`. Missing, duplicate, or nonzero-based rank reports fail materialization.

FSDP2 admission records:

- The module entry points that trigger FSDP hooks.
- The hook-entry policy.
- Sharding granularity.
- Whether a candidate bypasses hooks through direct submodule calls or patched methods.
- Bottom-up sharding order.
- In-place module mutations caused by sharding.
- All-gather, reduce-scatter, prefetch, and reshard settings.
- Mixed precision policy.
- Offload policy.

DTensor and tensor-parallel admission records:

- Device mesh.
- Input placements.
- Output placements.
- `dtensor_module_class`, admitted against the distributed policy's supported module classes.
- `to_local` gradient placement policy.
- `from_local` check policy.
- Uneven-shard handling.
- Async local tensor handling.
- Tensor-parallel output layout propagation.
- Sequence-parallel axis and output layout.
- Context-parallel axis and output layout.
- Context-parallel rotate method, either `all_gather` or `all_to_all`.
- `higher_order_diff_status`, with one admitted status for every input and output placement slot.

`tp.loss_parallel=true` requires exact cross-shard normalization for CE or KL losses and a multi-rank agreement check against the unsharded loss on the same logical batch.

Distributed executor API:

```python
def distributed_axis_manifest() -> AxisManifest: ...
def distributed_operation_factory(...) -> OperationFactory: ...
def distributed_reference_check(...) -> ReferenceCheck: ...
def distributed_materializer(...) -> Materializer: ...
```

The adapter must provide operation factories for:

- Starting ranks under `torchrun`.
- Initializing the declared process group.
- Building `DeviceMesh`.
- Creating DTensor placements.
- Applying `fully_shard` for FSDP2 and HSDP rows.
- Applying tensor-parallel plans with `parallelize_module`.
- Applying sequence-parallel and context-parallel execution.
- Redistributing DTensors at declared operator boundaries.
- Running collectives with declared overlap, prefetch, and bucket size.
- Reducing rank-local status and rank-local memory to global selection fields.

Distributed rows must run single logical batches. The adapter owns gather or placement conversion before comparison to single-device references.

## Numerical Thresholds

Reference checks receive explicit numeric thresholds from the operator spec, runtime reference check, or adapter policy. The package exposes this standard threshold table as `vptune.ext.STANDARD_THRESHOLDS`:

- `max_abs_diff`: `1e-4`.
- `max_rel_diff`: `1e-3`.
- `value_abs_diff`: `1e-4`.
- `multiply_max_abs_diff`: `1e-4`.
- `multiply_max_rel_diff`: `1e-3`.
- `inverse_max_abs_diff`: `1e-4`.
- `inverse_max_rel_diff`: `1e-3`.
- `inner_abs_diff`: `1e-4`.
- `inverse_residual`: `1e-4`.
- `symmetry_max_abs_diff`: `1e-4`.
- `psd_violation`: `1e-12`.
- `directional_abs_diff`: `1e-3`.
- `directional_rel_diff`: `1e-2`.
- `min_probe_norm`: `1e-3`. The dot-product and symmetry reference probes must have norm at least this value, so a zero probe cannot pass an identity vacuously.

Paired absolute and relative thresholds pass when either the absolute or relative scale is within tolerance. Unpaired thresholds pass directly.

Thresholds are fixed acceptance fields. They are not changed by a speed knob.
Rows that degrade reduction precision must provide the derived numeric error
bound fields from the feature manifest: $k$, $\epsilon$, $C_{\mathrm{op}}$,
$S_{\mathrm{row}}$, and the output norm floor. The derived absolute bound is
$B_{\mathrm{abs}}=C_{\mathrm{op}}\gamma_k(\epsilon)S_{\mathrm{row}}$ with
$\gamma_k(\epsilon)=k\epsilon/(1-k\epsilon)$ and $k\epsilon < 1$. The row fails
when measured error exceeds the derived bound or when the derived bound exceeds
the fixed acceptance threshold.

Operator-specific threshold policies:

- Gradient, JVP, VJP, and HVP declare denominators for directional checks.
- HVP declares the symmetry vector used for $\langle x,Hy\rangle = \langle y,Hx\rangle$ checks.
- GGNVP requires `symmetry_max_abs_diff`, `psd_violation`, and `inner_abs_diff` on the output-space loss Hessian, which is always a PSD metric.
- Inverse metric rows require `inverse_residual`, `symmetry_max_abs_diff`, and `psd_violation`; damped inverse rows also declare `damping_min` and `condition_number_max` thresholds.
- Metric inner-product rows require `inner_abs_diff` against the dense Gram reference; an `as_norm` row additionally requires the same-vector diagonal nonnegativity check (`psd_violation` on the diagonal entries), and an inverse-metric-inner `solve_then_reduce` row additionally requires the per-column `inverse_residual` and, when damped, `damping_min` and `condition_number_max`.
- FisherVP declares score policy and normalization denominator.
- Sampled FisherVP declares sample source, sample count, sample identity, normalization denominator, and sampling-bound formula when exact-Fisher comparison is enabled.
- Empirical FisherVP declares per-example loss reduction and normalization denominator.

## Saved Files

`vptune` writes JSON files and tensor files under a run directory selected by the caller when the selected row stores tensors.

```text
run_dir/
  candidates/
    <family>/<row_id>/candidate.json
  references/
    <family>/<row_id>/<check>.json
  full_size/
    <family>/<row_id>/result.json
  summaries/
    tuning.json
    selected_plan_validation.json
  tensors/
    <tensor_id>.pt
```

If more than one schema-valid row has the same family, row id, and check name path, later rows are written as numbered siblings next to the first file. Replay uses saved fields, not filename suffixes, to identify rows.

Every JSON record contains:

- Record type.
- Package version.
- Schema version.
- Operator spec fields.
- Input signature.
- Candidate settings.
- Candidate generator id and version.
- Adapter ids and versions.
- Target identity.
- Environment signature.
- Status.
- Measurements or error fields.
- Replay identity fields required for the record type.
- Schema-valid row content fields required for direct replay comparison.
- Materializer identity for selected-plan summaries.

Tensor files are referenced by tensor id, dtype, shape, and tensor identity metadata. JSON records never rely on implicit local state.

## Acceptance Tests

Package tests:

- Axis manifest contains every key and value in this spec, has one owner per key, assigns every key to one Class C group, applies merge rules deterministically, and rejects duplicate owners.
- Axis manifest rejects contradictory rows for packing, checkpoint and activation recompute, activation offload, compile aliases, SDPA kernels, DTensor placements, sampled Fisher, and exact categorical Fisher.
- Axis manifest rejects `compile.options.*=true` with `compile.mode` other than `None`, rejects `compile.mode=None` when all compile options are disabled, and uses `numeric.float32_matmul_precision` as the only CUDA matmul TF32 sweep key.
- Axis manifest rejects metric and inverse-metric rows whose path, block schedule, preconditioner, factor dtype, or factor residency is incompatible with the declared metric representation.
- `vp.problem(...)` and `vp.autotune(...)` reject composition specs because composition children require sibling products in a `TuningRun`.
- Operator constructors validate the closed-set fields of the typed `Loss`, `Output`, `Likelihood`, and `Metric` objects and raise at construction on an unsupported value; a typo never selects different math.
- Every manifest value has one admission rule and either one lowering rule or one adapter owner that supplies lowering.
- Gradient anchor matches direct autograd on a tiny MLP.
- JVP anchor matches finite difference.
- VJP anchor satisfies the dot-product identity.
- HVP anchor matches reverse-over-reverse and finite-difference gradient checks.
- Standard runtime builder runs gradient, JVP, VJP, and HVP from declared objectives and candidates.
- Standard runtime builder runs dense GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, metric, metric inner product, inverse metric, and inverse metric inner product candidates over full tensor trees.
- Standard runtime applies declared `dtype.parameter_storage`, `dtype.model_compute`, `numeric.float32_matmul_precision`, and `numeric.bf16_reduced_precision_reduction`; it preserves integer and boolean batch tensors.
- Grad-materialization tests cover tensor-tree returns for torch.func and eager `torch.autograd.grad` rows, `.grad` materialization for `backward_materialized_grad` rows, and rejection of rows that try to override the materialization derived from the AD path.
- Teacher-output tests cover CPU, pinned CPU, GPU, and recomputed teacher outputs, including equality rejection for recomputed teacher outputs that do not match the fixed teacher-output field.
- Numeric loss-scaling tests cover degree-one unscale, degree-two score-gradient unscale for FisherVP, sampled FisherVP, and empirical FisherVP, and rejection of rows that omit the exact unscale law.
- Fusion tests cover every `fusion.*` axis value, verify the registered fused subpath is actually called, require exact global normalization for fused CE and KL, and run the required higher-order agreement check for HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, and compositions containing them.
- Activation-offload tests cover CPU saved-tensor hooks and custom saved-tensor hooks, verify hooks execute on saved tensors, and reject offload rows whose hooks do not restore the same tensors for backward or higher-order AD.
- Layout tests cover `layout.vector_ops=foreach`, contiguity materialization, tied-weight alias preservation, active parametrization preservation, flat/per-layer/per-block reconstruction, and DTensor layout handoff to the distributed adapter.
- Standard runtime rejects registered axes whose execution belongs to adapters.
- `vhp` candidate path reports an HVP result and requires symmetry plus finite-difference directional checks.
- GGNVP cross-checks dense `J^\top H Jv` against the independent JVP-Hessian-VJP anchor on a tiny model.
- GGNVP enforces PSD on the output-space loss Hessian: it rejects mismatched output-Hessian shape, nonfinite loss Hessians, nonsymmetric loss Hessians, indefinite loss Hessians, and missing dot-product checks. A non-PSD output Hessian, whether from a non-convex `vp.loss.from_scalar(...)` or from `vp.loss.declared_psd(...)`, fails admission with "GGN requires a PSD output-space metric."
- FisherVP anchor computes exact score-gradient outer products from the declared per-example score objective and rejects mismatched precomputed matrices.
- Exact categorical NLL Fisher is expressed by GGNVP and covered by GGNVP dense and JVP-Hessian-VJP tests.
- Sampled FisherVP uses declared fixed sample table or fixed seed and sample count, repeats exactly for the same source, rejects mismatched sampled-score matrices, and runs exact-Fisher comparison only when a sampling-bound formula is declared.
- EmpiricalFisherVP anchor computes per-example-gradient outer products from the declared per-example loss objective and rejects mismatched precomputed matrices.
- EmpiricalFisherVP standard runtime has both loop and `vmap(grad)` per-example-gradient paths, and `vmap_chunk_size` is honored only on the vmap path.
- Standard dense metric materialization returns one object with metric multiply, inverse multiply, and metric inner product. Materialized `metric_vp` defaults to multiply; materialized `inverse_metric_vp` defaults to inverse multiply.
- Metric tests cover dense, diagonal, block-diagonal, KFAC, low-rank, and GGN-derived metrics; every factored metric supplies its required fields through the typed `Metric`, reconstructs a dense reference from those fields, and matches dense multiply, solve, and inner-product references.
- Inverse-metric tests cover every solve path, every preconditioner, factor reuse, block schedules, and rejection of `inverse_metric.iteration_budget` on direct solve rows.
- Composition reference checks run child operator anchors and write child reference rows linked by ordered child reference descriptors from the parent row.
- Composition tests declare children through `vp.composition(..., children=..., combine=...)`, derive dependencies from the child-name leaves of the `combine` expression, reject any separately supplied composition dependency list, and cover `selected_child_rows`, `inline_child_lowering`, `validate_each_child`, and `validate_composed_output`.
- Composition combinator tests cover `vp.compose` lowering to `sequential_composition`, `vp.linear_combination` lowering to `linear_combination` with the declared coefficients and a `vp.scaled_identity` leaf, `vp.source` seeding a vector-valued composition called as `composition(batch)`, nested expressions, rejection of a `vp.source` in a non-seed position, and rejection of shape-incompatible `vp.compose` adjacencies and `vp.linear_combination` terms.
- Positive-definiteness tests reject `inverse_metric_vp(vp.metric.matrix_free(operator=ggn), damping=vp.damping.scalar(0.0))` on a `conjugate_gradient`, `cholesky_solve`, `eigh_solve`, or `svd_solve` row, and accept it for `vp.damping.scalar(lam)` with `lam > 0`.
- EKFAC metric tests cover `vp.metric.ekfac` multiply and `factorized_solve` in the Kronecker eigenbasis against a dense reference.
- Square-root tests cover `sqrt_metric_vp` and `inverse_sqrt_metric_vp` for KFAC, EKFAC, dense, block, and matrix-free metrics, and verify the factor round-trip $L(L^\top v)$ matches $Mv$ on a reference (the symmetric $M^{1/2}(M^{1/2}v)$ only on an eigenbasis path).
- Metric inner-product tests cover `metric_inner_vp` and `inverse_metric_inner_vp` for dense, KFAC, EKFAC, low-rank, GGN-derived, and matrix-free metrics: the $k\times k$ Gram matches the dense reference for `multiply_then_reduce`/`solve_then_reduce`, `factored_gram`, and `sqrt_apply_reduce`; a block right-hand side matches the single-column Gram; an `as_norm=True` row admits only `sqrt_apply_reduce`, its same-vector diagonal is nonnegative, and it rejects `multiply_then_reduce`, `solve_then_reduce`, and `factored_gram`.
- Per-example gradient tests cover `vp.per_example_gradient` against a loop reference and confirm `empirical_fisher_vp` equals its outer-product reduction.
- Typed damping tests cover `vp.damping.scalar`, `vp.damping.per_group`, and `vp.damping.kfac_pi`.
- Solver-tolerance tests cover `tol` on an iterative inverse row (`conjugate_gradient`) setting both the CG stopping criterion and the accepted inverse residual.
- Cohort-input tests pin `layout.vector`/`dtype.vector` across several products through `cohort_constraints` and reject a plan whose products disagree on the pinned axes.
- Multi-RHS tests cover `inverse_metric.multi_rhs=block` against the single-column solve, and `Operator.bind(batch=...)` producing a `(vector,)`-arity callable whose compiled row reuses across calls.
- Typed-object validation tests reject a typo in any closed-set field of `Loss`, `Likelihood`, `Metric`, `SampleSource`, and `Damping`, and reject a `from_scalar`/`declared_psd_matrix_free` loss whose probe $H_\ell$ is non-PSD.
- Replay tests cover the new identity fields: a `matrix_free` metric distinguished by its wrapped product, a `from_scalar` callable distinguished by its value version, a composition distinguished by a `linear_combination` coefficient, a sampled Fisher distinguished by its sample seed, an inner product distinguished by its `as_norm` setting, and an iterative inverse row distinguished by its `tol` value.
- KFAC metric multiply, inverse, and inner product match dense references.
- Metric and inverse-metric checks reject nonsymmetric and indefinite dense metrics.
- Threshold logic covers over-threshold failure, abs-or-rel passing, derived numeric error bounds, zero-denominator relative error, and nonfinite values.
- Compile tests cover disabled eager rows, callable boundaries, fullgraph graph-break rejection, compiled autograd, CUDA graph rows, max-autotune normalization, compile amortization, cold compile, warm cache, and recompile counts.
- Attention tests cover every frontend listed in the manifest, every SDPA kernel listed in the manifest, priority ordered SDPA kernels, `sdpa_kernel` context entry, `set_attn_implementation`, registered attention and mask functions, packed exact attention, blockwise exact attention, and full-size backend-triggering checks.
- Attention executor tests cover a non-Transformers module using `pytorch_sdpa_direct`, `patched_eager`, `packed_exact`, `blockwise_exact`, SDPA kernels, partitioning, and padding through a model-attention-location descriptor.
- Distributed tests cover every distributed axis, process-group creation, `DeviceMesh`, DTensor placements, FSDP2 `fully_shard`, HSDP mesh dims, tensor-parallel plans, sequence parallel, context parallel, DTensor redistribution, collective overlap, global memory reduction, rank failure propagation, and multi-rank reference agreement.
- Search tests cover `admission`, `smoke`, `fast`, `balanced`, `thorough`, and `exhaustive` strategies on bounded candidate sets.
- Autobatch tests cover every integer-domain field, domain admission, value-to-settings mapping, reference-before-probe behavior, failure frontier termination, and replay from selected observed values.
- JSON schema validation rejects stale direct identity fields, stale input signatures, stale thresholds, stale generator versions, stale metric-representation replay fields, and stale dependency selections.
- Saved reference and full-size rows carry schema-valid content fields. Replay compares the saved row fields used by selection and materialization directly against the loaded rows and recomputed selected plan.
- Saved-run replay materializes the selected plan without running search.
- Memory stability rejects post-call reserved growth.
- Measurement records required memory fields, OOM rows, runtime failure rows, reference runtime failure rows, timing policy, probe-call exclusion, backend release after failed calls, and long-row single-call behavior.
- Selection chooses lower memory within the near-fastest band.
- Selection tests cover cohort comparison by summed selected row scores for eager, compiled, distributed, and compiled distributed rows.
- Dtype coherence is expressed through a `CohortConstraint`, not through a dtype-only selector.
- Cohort selection supports generic single-key and multi-key cohort constraints, covered-family subsets, dependency-aware assignment search, and replay from saved rows.
- Blocked descendants write `PrerequisiteFailed` full-size rows.
- Failed rows from non-selected cohort assignments remain in returned plans when another cohort assignment completes.
- Plan replay rejects stale target, runtime, adapter, materializer, candidate-row, dependency-identity, and cohort-assignment identity.
- Selection rejects stale signatures and all-failed families.
- `functional_call` tests cover tied weights, parametrizations, buffers, module mode, in-place writes, and direct runtime admission.
- `torch.func` tests cover every torch.func-transform path in the manifest, including gradient, JVP, VJP, HVP, GGNVP JVP/VJP, FisherVP, sampled FisherVP, and empirical FisherVP paths; they also cover `vmap` randomness, dynamic-shape rejection, `.item()` rejection, forward AD coverage failure, and direct runtime admission.
- Checkpoint tests cover RNG preservation and deterministic recomputation.
- Transformer adapter tests cover model identity, eager, SDPA, FlashAttention, FlexAttention, paged attention, registered Transformers attention, admission setting ownership, `output_attentions=True` rejection, softcap signatures, mask semantics, dropout policy, and full-size agreement gates for non-math Transformers attention rows.
- Distributed adapter tests cover rank agreement, global max memory, per-rank failure propagation, FSDP hook entry, FSDP policy axes, admission setting ownership, DTensor gradient placement, and mode-specific layout admission for tensor, sequence, and context parallel rows.
- Selected-plan validation follows stored family order, receives materialized dependency context, writes per-family validation rows named `selected_plan_validation`, writes `summaries/selected_plan_validation.json`, records validator identities in the plan, fails the selected settings when any selected family fails, and replay rejects missing, forged, stale, or failed validation rows and summaries.
- Root imports expose the user-facing surface (model builders, the typed math objects, the operator constructors, `space`, `search`, `cuda`, `tune`, and the lower-layer `problem`, `autotune`, and replay functions) and never `Candidate`, `RuntimeConfig`, or the axis machinery; adapter helpers are available through `vptune.adapters`, and extension helpers are available through `vptune.ext`.

Pilot adapter tests:

- Each pilot family can be expressed as a `vptune` family.
- Pilot selected settings can be produced from a `vptune` plan.
- Pilot selected-plan validation passes from the produced settings.
- Downstream pilot stages accept the produced readiness state.

## Package Layout

Package shape:

```text
vptune/
  SPEC.md
  FEATURES.md
  SCRATCHPAD.md
  Makefile
  pyproject.toml
  src/
    vptune/
      __init__.py
      py.typed
      admission.py
      anchors.py
      autobatch_bridge.py
      candidates.py
      checks.py
      data.py
      errors.py
      identities.py
      attention.py
      io.py
      measure.py
      operators.py
      reference.py
      runtime.py
      select.py
      selection_core.py
      cohorts.py
      run.py
      schemas.py
      tensor_tree.py
      adapters/
        transformers.py
        distributed.py
        pilot.py
  tests/
    test_core.py
    test_attention_executor.py
    test_standard_runtime.py
    test_transformers_adapter.py
    test_distributed_adapter.py
    test_pilot_adapter.py
```

## Implementation Order

1. Axis manifest with every key, value domain, owner, Class C group, merge rule, admission rule, and lowering owner from this spec.
2. Core data classes, JSON schemas, replay fields, and direct-field replay checks.
3. Standard runtime lowering for gradient, JVP, VJP, HVP, GGNVP, FisherVP, sampled FisherVP, empirical FisherVP, per-example gradient, metric multiply, metric inner product, metric square-root multiply, inverse metric multiply, inverse metric inner product, inverse metric square-root multiply, and composition.
4. Package-owned anchors and full-size gates.
5. Measurement, memory sampling, failure rows, selection, and shared selector reuse for tuning and replay.
6. Search strategies: `admission`, `smoke`, `fast`, `balanced`, `thorough`, and `exhaustive`.
7. Autobatch domains for integer axes.
8. Core attention executor: PyTorch SDPA context, direct SDPA calls, patched eager attention, packed exact attention, blockwise exact attention, partitioning, padding, and model-attention-location descriptors.
9. Transformers executor: model loading, `set_attn_implementation`, Transformers FlashAttention rows, Transformers FlexAttention rows, paged rows, registered Transformers attention, and model-attention-location descriptors for the core attention executor.
10. Compile executor: `torch.compile`, compiled autograd, CUDA graph rows, max-autotune rows, call-horizon scoring, and recompile measurement.
11. Activation and memory executor: checkpointing, manual recompute, saved-tensor hooks, residency movement, and output buffers.
12. Layout executor: flat, per-layer, per-block, per-shard, DTensor, foreach vector ops, contiguity, aliasing, and parametrizations.
13. Distributed executor: process groups, `DeviceMesh`, DTensor placement, FSDP2, HSDP, tensor parallel, sequence parallel, context parallel, communication scheduling, and rank-global measurement.
14. Pilot adapter lowering and readiness conversion.

The package is complete when every axis value in this spec has executable lowering or adapter lowering, every required check runs, and the full test list passes.
