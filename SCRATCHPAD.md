# Scratchpad
Branch: codex-implement-spec @ 38d7f77

## TODO
- [x] Read SPEC.md in full.
- [x] Read FEATURES.md in full.
- [x] Read FAILURE_MODE_ANALYSIS.md in full.
- [x] Read REPO_AUDIT.md and extract mistakes to avoid.
- [x] Inspect current package and tests before editing.
- [x] Run local lint and tests for current source/test changes.
- [x] Run hardware-gated tests on Ferranti.
- [x] Audit remaining spec gaps from current evidence.
- [x] Derive the next missing spec step and implement it vertically.
- [x] Run lint and tests required for edited files.
- [x] Review current production-readiness guidance and map it to this repo.
- [x] Complete code review for API stability, failure modes, and maintainability.
- [x] Fix production-readiness gaps found by review.
- [x] Run real user workflows against the public API.
- [x] Check README examples against the final API.
- [x] Verify build/install from a fresh environment.
- [x] Review test warnings and decide whether they require code changes.
- [x] Decide handling for untracked audit docs and scratchpad state.
- [ ] Rerun lint, tests, and any required hardware checks after changes.

## Open questions for the user

## Uncertain / ideas to explore
- Need verify every audit finding against current files before acting on it; the audit itself says live edits superseded some findings.
- Prefer adding rows to existing tables and parametrized tests over new per-case functions.
- Need stop and raise any conflict where current code is better than SPEC.md or where I disagree with the spec.
- Metadata was moved to version 1.0.0 and `Development Status :: 5 - Production/Stable`; `PACKAGE_VERSION` and `uv.lock` were updated with it.
- README now has a root public API quickstart that tunes built-in softmax-cross-entropy gradient on CPU.
- Audit docs stay untracked local review inputs and are excluded from package output. `SCRATCHPAD.md` stays the tracked live scratchpad required by repo instructions and is also excluded from package output.
- Best-practice mapping used NIST SSDF, PyPA packaging/build docs, Hatch sdist file selection docs, and OpenSSF Scorecard checks.

## Notes to self
- Existing scratchpad was explicitly ignored for this session.
- SPEC Implementation Order starts with manifest parity, then core data/schema/replay, then runtime lowerings, anchors, measurement/selection/search, autobatch, attention, adapters, compile, memory, layout, distributed, pilot.
- FEATURES permits per-value ownership only for attention.frontend; all other keys have exactly one owner.
- No fallback behavior is allowed. Unsupported combinations should be explicit admission failures or explicit raises at the required layer.
- REPO_AUDIT run 3 flags current work to verify: composition algebra lowering, missing loss constructors, manifest meta-tests, admission timing, alias normalization, with matrix_free status treated as possibly changed.
- FAILURE_MODE_ANALYSIS warns against copying each operator/value into separate functions when one data-driven dispatch or parametrized test covers the behavior.
- Current tree already has loss constructors, composition algebra tests, and manifest meta-tests that REPO_AUDIT listed as missing.
- Local `make lint-fix` passed.
- Local `make test` passed with 860 passed, 11 skipped, 38 warnings after escalation for uv cache access.
- Ferranti hardware-gated job `398465` on `h100-ferranti` passed 8 tests in 10.89s with two H100 GPUs: CUDA SDPA backends, single-rank and two-rank NCCL, GPU vector residency, GPU input movement, and GPU teacher outputs.
- Ferranti pinned-memory job `398466` on `h100-ferranti` passed 3 tests in 4.16s with one H100 GPU: pinned vector residency, pinned input movement, and pinned teacher outputs. All 11 local skipped hardware/backend tests now have Ferranti pass evidence.
- Commit `17003cc` pushed to origin/codex-implement-spec for cluster testing.
- Meta-audit command passed: 6 tests covering manifest-vs-SPEC domains, manifest-vs-FEATURES keys, manifest value/check coverage, acceptance-test coverage, descriptor identity fields, and root import boundary.
- `BLOCKERS.md` currently contains `None`.
- User approved editing SPEC.md to add missing `src/vptune/ext.py` to Package Layout because the public API already requires `vptune.ext`.
- User approved keeping `public.py`; SPEC Package Layout now lists `public.py` and the real test files.
- Root `vp.materialize` now matches SPEC with only `name=`, and tests/internal calls use `name=` instead of `family=`.
- Verification after edits: `make lint-fix` passed; focused materialize/layout tests passed 3; full `make test` passed with 860 passed, 11 skipped, 38 warnings.
- Current-tree verification before final Ferranti run: `make lint-fix` passed; focused manifest/acceptance/API tests passed 7; full `make test` passed with 860 passed, 11 skipped, 38 warnings.
- Final commit for code/spec/test verification is `38d7f77`.
- Ferranti final pinned-memory job `398473` on `h100-ferranti` passed 3 tests in 4.90s with one H100 GPU.
- Ferranti final hardware-gated job `398472` on `h100-ferranti` passed 8 tests in 10.96s with two H100 GPUs: CUDA SDPA backends, single-rank and two-rank NCCL, GPU vector residency, GPU input movement, and GPU teacher outputs.
- Production-readiness pass started after acceptance completion. Current concrete gaps: README only has title and one sentence; `pyproject.toml` still declares `Development Status :: 3 - Alpha`; fresh install and public workflow examples still need this-pass verification.
- A CE gradient tuning example with an all-ones reference vector failed because the directional reference denominator was zero, producing nonfinite `directional_rel_diff`; a nondegenerate vector fixes the example and the code path passes.
- Added `tests/test_public_api.py::test_public_tune_builtin_softmax_cross_entropy_gradient`.
- Added `tests/test_core.py::test_package_version_matches_project_metadata`.
- Verification after README/version edits: `make lint-fix` passed; focused metadata and public CE-gradient workflow tests passed 2; README workflow command passed and printed the expected gradient tensor.
- First package build succeeded but sdist included `.uv-cache`, `SCRATCHPAD.md`, `FAILURE_MODE_ANALYSIS.md`, and `REPO_AUDIT.md`. Added explicit Hatch sdist include list.
- Rebuilt package at `/private/tmp/vptune-prod-dist-20260607-2`: wheel contains only installable `vptune` package and dist-info; sdist contains source, tests, design docs, README/LICENSE/pyproject/Makefile/uv.lock, and omits cache, scratchpad, and audit docs.
- Fresh wheel install into `/private/tmp/vptune-prod-venv-20260607-2` succeeded with `vptune==1.0.0`, `torch==2.12.0`, and pinned `autobatch` commit `a0663ac586c61c4593e9b9c6031f43f464f61139`.
- Fresh installed README workflow passed and printed package version `1.0.0` plus the expected gradient tensor.
- `uv pip check` passed in the fresh environment; `python -m pip check` is unavailable because the uv-created venv has no `pip` module.
- `pip-audit` over the fresh site-packages found no known vulnerabilities in PyPI packages. It skipped `autobatch` and `vptune` because direct/local packages are not in PyPI advisory matching.
- Warning review: full local test suite has 38 warnings, all from PyTorch internals: 18 `torch.jit.script` deprecations in attention tests, 14 `torch.jit.script_method` deprecations in standard runtime tests, and 6 `torch.fx` const-fold UserWarnings. Fresh install also shows PyTorch's no-NumPy warning when NumPy is absent. No package code calls `torch.jit.script` or `script_method`; adding NumPy solely to silence PyTorch would add an unused dependency.
- Final local lint gate after package edits: `make lint-fix` passed and `git diff --check` passed.
