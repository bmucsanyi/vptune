# Scratchpad
Branch: codex-implement-spec @ d2d3d99

## TODO
- [x] Read SPEC.md in full after latest binding reminder: 2361 lines.
- [x] Read FEATURES.md in full after latest binding reminder: 1293 lines.
- [x] Read FAILURE_MODE_ANALYSIS.md in full after latest binding reminder: 93 lines.
- [x] Read REPO_AUDIT.md in full after latest binding reminder: 151 lines.
- [x] Update SPEC to make KFAC damping exclusively typed-operator damping and reject metric-owned declarations.
- [x] Lower `vp.damping.per_group(...)` for dense, low-rank, GGN-derived, and matrix-free metrics using parameter-surface group names.
- [x] Run focused per-group tests after lint cleanup.
- [x] Run `bash -lc 'make lint-fix'`.
- [x] Run `bash -lc 'make test'`.
- [x] Commit and push the current verified tree for Ferranti hardware testing.
- [x] Pull the pushed branch in `/home/hennig/hmx900/repos/vptune` on Ferranti.
- [x] Submit the hardware-guarded tests to the `h100-ferranti` partition and record the result -- Slurm job 398414.
- [x] Re-audit remaining blockers against live code after the goal continuation.
- [x] Run focused manifest, FEATURES parity, acceptance-coverage, and root-surface tests.
- [x] Run focused boundary tests for the remaining blocker cases.
- [x] Finish sampled-Fisher run-level admission for the two named sampling-bound formulas.
- [x] Finish matrix-free preconditioner runtime test with `inverse_metric.preconditioner_product`.
- [x] Add inverse-sqrt matrix-free Lanczos `tol` runtime test.
- [x] Update `BLOCKERS.md` after the three resolved implementation gaps are verified.
- [x] Run focused tests for the current slice -- 11 passed.
- [x] Run `bash -lc 'make lint-fix'` -- passed.
- [x] Run `bash -lc 'make test'` -- 858 passed, 11 skipped, 38 warnings.
- [x] Resolve the remaining loss-constructor assumption blocker by making the executable semantics explicit in SPEC.
- [x] Clear `BLOCKERS.md` after the SPEC loss clarification.
- [x] Run final `bash -lc 'make lint-fix'` -- passed.
- [x] Run final `bash -lc 'make test'` -- 858 passed, 11 skipped, 38 warnings.
- [ ] Rerun Ferranti hardware tests after the final local tree is ready for full DoD.

## Open questions for the user
- None.

## Uncertain / ideas to explore
- REPO_AUDIT guard: do not add another near-identical branch or spy test. Prefer one dispatch point and parametrized behavior tests.

## Notes to self
- Ferranti for this goal is `ssh -i ~/.ssh/slurm_tue hmx900@134.2.168.205 -p 2221` only.
- No `owl569` access under any circumstance in this session.
- The latest binding read completed on 2026-06-07 before resuming implementation.
- Current SPEC says sampled Fisher public sources accept `sampling_bound=`, with `abs_or_rel`, `matrix_bernstein`, and `hutchinson_relative_variance`.
- Current SPEC says inverse-sqrt `tol` is admitted only for matrix-free Lanczos and sets the accepted Lanczos residual estimate.
- Current SPEC says `inverse_metric.preconditioner=matrix_free` requires companion setting `inverse_metric.preconditioner_product`.
- Current SPEC says KFAC damping is only through typed `vp.damping.*`; metric-owned KFAC damping declarations reject.
- Current SPEC now spells out KL, MSE, declared-PSD, and declared-PSD matrix-free loss semantics to match the executable code and tests.
- Existing Ferranti evidence before this final tree: Slurm job 398414 passed 12 hardware tests on `h100-ferranti`.
