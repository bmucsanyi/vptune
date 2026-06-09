# Scratchpad
Branch: codex-implement-spec-v2 local @ 3770a08 = origin (user switched local branch to v2)

<!-- writing-rules: justified -->
writing-rules justification: filler-adjectives - the word appears only inside the verbatim title of git commit 68a7c05, quoted to identify that commit; rewording would misname the commit.

## TODO
- [x] Narrow plan items 1-5 + five-layer restructure (704c04b) + layering meta-test + SPEC layout amendment.
- [x] GPU: 399306 (2898688) 899/1 PASS; 399380 (9bc8204) 899/1 PASS; 399381 (704c04b) submitted - poll at the END per user.
- [x] Mapping extensions for 14 bullets with verified existing coverage (+31 names).
- [x] Eigenbasis sqrt: spec divergence found by round-trip test; now symmetric AND factored apply (no U S U^T materialization); pinning test updated (bd8869a).
- [x] New behavioral tests: sqrt round-trips + block sqrt; matrix-free 4-solve-path admission (2248); cohort pinning layout.vector/dtype.vector + disagreement rejection (2255); replay identity six fields (2258); exhaustive strategy (2266); fullgraph graph-break rejection with dynamo reset (2262) (c99a6e3..bf65583).
- [x] Real-model tests (3770a08): transformer (2-layer encoder, padding masks) + conv net through public API: gradient/hvp vs autograd (float64 exact), GGN symmetry+PSD, categorical fisher->GGNVP routing assertion, per-example rows compose empirical fisher (/N convention), sampled fisher fixed-seed repeat. Findings: CPU flash-SDPA lacks double-backward AND forward AD -> tests pin SDPBackend.MATH via sdpa_kernel.
- [ ] Meta-test strengthening: (a) mapped tests must contain a behavioral assertion (Assert node, assert_close, or raises with match=) via AST; (b) manifest_value_literal_covered searches test-function bodies only (test_core.py:781 area).
- [ ] RuntimeValueError(MaterializationError) wiring for engine/runtime_values.py raises + errors.py + exports + docstring updates.
- [ ] Remaining partial-gap tests (next batch): 2241 vmap_chunk_size honored on vmap path; 2227 materialization-override rejection; 2231 offload hooks restore-wrong-tensors rejection; 2236 PSD enumerations (shape/nonfinite/dot-check + exact message dense path); 2239 sampled exact-comparison gating on declared bound; 2254 Lanczos tol residual on matrix-free inverse-sqrt; 2251 inner products KFAC/low-rank/GGN-derived/matrix-free + factored_gram.
- [ ] VERIFICATION.md (UNTRACKED): bullet-by-bullet table with grades and what changed this session.
- [ ] Final: local gates + push + cluster checkout final sha + full-suite GPU job + poll 399381 and the final job; certification report per DoD (local count; GPU count+job id+commit; parity results; BLOCKERS entries).

## Open questions for the user
- None pending.

## Uncertain / ideas to explore
- vp.likelihood.categorical fisher_vp rejects with "exact categorical Fisher is represented by GGNVP" (spec-conformant routing; asserted in real-model test).
- pre-commit format hook can block commits: run make lint-fix before git commit.

## Notes to self
- Module paths: vptune.core.{data,identities,operators,tensor_tree}; vptune.axes.{candidates,admission}; vptune.engine.{runtime,runtime_values,derivatives,ggn,fisher,metrics,vectorization,composition,layout,memory,compile,attention,anchors,checks,reference}; vptune.tuning.{run,measure,select,selection_core,cohorts,autobatch_bridge,schemas,io}.
- ACCEPTANCE_TEST_COVERAGE dict at tests/test_core.py:294 keyed by bullet-prefix; extend via regex inserter (keys are short prefixes, check exact key text first).
- Test scaffolds: metric ops via standard_operation_result + representation helpers (dense/block/kfac at tests/test_standard_runtime.py:814+); strategy tests via cpu_target + SearchPolicy(strategy=...) + TwoProbeData/TwoVectorProvider; cohort tests via recorded_tuning_problem + paired SequenceClock ticks (mirror green tests exactly); compile via compile_settings()+compiled_boundary_operation; parametrized fullgraph tests need torch._dynamo.reset() (cache pollution across params).
- Empirical fisher convention: rows.T @ (rows @ v) / N. CE default denominator num_tokens (mean for per-example classification).
- GPU: ssh -i ~/.ssh/mlcloud_auth -p 2221 owl569@134.2.168.205; /weka/hennig/owl569/repos/vptune-verify; .torch-overlay torch 2.12.0+cu130; hw_gate.sbatch full suite 2x h100.
- Local counts now: 906 passed, 11 skipped, ~36s. Commit style: regular messages, no co-author. make lint-fix before commit (pre-commit hook).
