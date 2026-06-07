# Scratchpad
Branch: codex-implement-spec @ 17f1eea

## TODO
- [x] Read SPEC.md in full after latest binding reminder.
- [x] Read FEATURES.md in full after latest binding reminder.
- [x] Read FAILURE_MODE_ANALYSIS.md in full after latest binding reminder.
- [x] Read REPO_AUDIT.md in full after latest binding reminder.
- [x] Lower `vp.damping.per_group(...)` for dense, low-rank, GGN-derived, and matrix-free metrics using parameter-surface group names.
- [x] Shorten the long per-group public test name for lint.
- [x] Remove stale per-group blocker text from BLOCKERS.md.
- [x] Run focused per-group tests after lint cleanup.
- [x] Run `bash -lc 'make lint-fix'`.
- [x] Run `bash -lc 'make test'`.
- [ ] Commit and push the current verified tree for Ferranti hardware testing.
- [ ] Pull the pushed branch in `/home/hennig/hmx900/repos/vptune` on Ferranti.
- [ ] Submit the hardware-guarded tests to the `h100-ferranti` partition and record the result.
- [ ] Continue any newly unblocked SPEC implementation gaps once a spec/API field exists.

## Open questions for the user
- SPEC/API decisions are needed before full DoD can hold: sampled-Fisher sampling-bound formula field, KFAC metric-owned `dampings`, `inverse_sqrt_metric_vp(..., tol=...)` residual semantics, sibling-product field for `inverse_metric.preconditioner=matrix_free`, and public distributed-space required-field derivation.

## Uncertain / ideas to explore
- Rejected wiring `inverse_metric.preconditioner=matrix_free` for now: SPEC requires a named sibling preconditioner product, but the current public API and candidate row expose no product-name field for that setting.
- Rejected adding sampled-Fisher exact-comparison public fields for now: SPEC says `SampleSource` carries a sampling-bound formula, but the documented constructors are only `fixed_seed(seed, count)` and `table(table=..., identity=...)`.
- Rejected metric-owned KFAC dampings for now: SPEC names `vp.metric.kfac(..., dampings=None)` but defines executable damping through `vp.damping.scalar`, `vp.damping.per_group`, and `vp.damping.kfac_pi`.
- FAILURE_MODE_ANALYSIS and REPO_AUDIT process guard: before adding another near-identical branch, test, or validator, collapse to one dispatch table or parameterized behavior test. Code size is part of acceptance.

## Notes to self
- Ferranti for this goal is `ssh -i ~/.ssh/slurm_tue hmx900@134.2.168.205 -p 2221` only.
- No `owl569` access under any circumstance in this session.
- The active goal text explicitly lifts the local git restriction for Ferranti sync; use git only for the required commit/push/pull path and do not touch `owl569`.
- Full re-read completed on 2026-06-07: SPEC.md 2361 lines, FEATURES.md 1293 lines, FAILURE_MODE_ANALYSIS.md 93 lines, REPO_AUDIT.md 151 lines.
- Repeated full re-read completed after the latest binding gate on 2026-06-07 before resuming lint fixes.
- `vp.damping.per_group(...)` now lowers for diagonal, block-diagonal, KFAC, EKFAC, dense, low-rank, GGN-derived, and matrix-free typed metrics. Dense, low-rank, GGN-derived, and matrix-free use parameter-surface names and range identity.
- Matrix-free per-group inverse and inverse-inner are covered through a public `vp.tune(...)` run that binds a GGN sibling product and checks dense reference behavior.
- Focused per-group tests passed after the latest binding re-read: 3 passed, 18 warnings.
- Lint found stale type annotations only: EKFAC leaf damping is scalar after group resolution, and CG batch solve receives scalar or per-group damping.
- `make lint-fix` passed after the per-group dense/low-rank/GGN-derived/matrix-free damping slice.
- `make test` passed after the per-group dense/low-rank/GGN-derived/matrix-free damping slice: 852 passed, 11 skipped, 38 warnings.
- Existing local evidence before this per-group slice: `make lint-fix` passed and `make test` passed with 850 passed, 11 skipped, 38 warnings.
- Last hardware evidence: Ferranti Slurm job 398391 on `h100-ferranti` completed with 12 passed in 14.21s and exit code 0:0.
- Hardware rerun command should target CUDA SDPA backend tests, GPU vector/input/teacher residency tests, and gloo/NCCL process-group tests.
