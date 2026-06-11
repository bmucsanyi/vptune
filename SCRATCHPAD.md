# Scratchpad
Branch: codex-implement-spec-v2 local @ 8cbedf3 + enumeration-tail batch (uncommitted)

## TODO
- [x] All prior milestones (narrow plan, conversion, five layers, real models, meta-tests, error domains) per earlier records.
- [x] GPU job 399387 at 8b13435: COMPLETED 0:0, 924 passed / 1 skipped in 82s.
- [x] Enumeration-tail batch, all bullets:
  - 2213 owner/group uniqueness + duplicate-owner rejection + deterministic merges (3 new tests).
  - 2215 compile option/mode combination rejections + only-TF32-key (1 new test).
  - 2257 metric-kind production gate in public.py _typed_metric_operator + typo rejections test (Metric/Likelihood/Damping).
  - 2235 dense GGNVP cross-check: engine dense_anchor_errors == 0 + public independent jacobian/hessian product (2 new tests).
  - 2251 matrix-free inner execution: single + block k x k Gram vs dense reference (1 new test).
  - 2256 bind reuse: isolated post-tune call asserts one call event, no compile (strengthened existing test).
  - 2230 real-rewriter higher-order agreement: HVP fused-module pass/broken-fail + GGN/Fisher trio fused rows through reference check (1 new test).
  - 2268 staleness kinds isolated (signature, generator version, dependency selection, metric-representation field) (1 new test).
  - 2269 saved-row schema validity + replay-from-disk agreement (1 new test).
  - 2272 OOM rows with captured samples + reference runtime failure rows (2 new tests).
  - 2274 distributed + compiled-distributed cohort sums with horizon flip (1 new test).
  - 2276 single-key constraint remap (dtype test, same generic machinery).
  - 2279 dependency-identity replay staleness (1 new test) + cohort-assignment remap.
  - 2280 all-failed family rejection with pinned message (1 new test).
  - 2282 every torch.func path: remapped traced execution tests (core paths, GGN variants, fisher trio score paths).
  - 2263/2264 attention remaps (mask formatter, packed/blockwise, segmented partition, executor rejections, CUDA kernel agreement, full-size check).
  - 2265 distributed remaps (13 traced per-item tests; NCCL rows on GPU gate).
  - 2284 flash/flex/paged implementation-selection parametrized test (7 rows) + admission/dropout remaps.
- [x] OperatorExecutionError judgment: no new class; spec error list closed; OperationMeasurementError under MeasurementError is the placement (measure.py:164).
- [x] Local: 937 passed, 11 skipped; make lint-fix green.
- [x] VERIFICATION.md: genuine 78, partial 1 (2219 meta-test proxy), name-only 0; prior count line (65/14) was stale vs its own table (had 20 partial rows).
- [ ] Commit + push, run hw_gate.sbatch on Ferranti, record job id + counts. BLOCKED: the permission system rejected git commit; the standing no-git rule needs the user's explicit lift in this session (the prior-session authorization in CODEX_SESSION_ANALYSIS.md does not carry).

## Open questions for the user
- Authorize git commit+push of this batch (or commit yourself), then I run the Ferranti hw gate on the new commit.
- 2219 is the lone partial bullet: is the strengthened meta-test proxy acceptable, or do you want a value-by-value behavioral audit of all ~180 axes?

## Notes to self
- Stateful module path restricted to gradient/vjp/hvp (runtime_values.py:2218); GGN/Fisher fused-row agreement runs against declared-objective anchors; anchors strip fusion.*/call.* (anchor_settings, runtime.py:2587).
- select_cohort requires passed rows per assignment; a failed row crashes in median_elapsed_seconds. The tune flow filters failed rows before assignments are built, so this is the callers' precondition, no production gap.
- Metric-kind gate: vp.metric_vp previously accepted typo'd kinds into factorized defaults. _METRIC_KINDS in public.py before _typed_metric_operator.
- Matrix-free engine idiom: standard_runtime_with_matrix_free_bindings(standard_runtime_config(...), bindings={name: callable}, ...).operation_factory.
- 2219 is the lone partial: per-value behavioral trace vs strengthened meta-test proxy. Raising it would need a value-by-value audit (~180 axes); judgment for the user whether the proxy suffices.
- GPU jobs: 399306/399380/399381/399385/399387 all COMPLETED 0:0; hw_gate.sbatch at ~/repos/vptune-verify (owl569), conda env unlearning.
- Commit style: regular messages, no co-author; make lint-fix before commit. Git+Ferranti authorized for this branch per CODEX_SESSION_ANALYSIS.md binding override.
