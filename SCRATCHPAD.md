# Scratchpad
Branch: codex-implement-spec @ 17003cc

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

## Open questions for the user

## Uncertain / ideas to explore
- Need verify every audit finding against current files before acting on it; the audit itself says live edits superseded some findings.
- Prefer adding rows to existing tables and parametrized tests over new per-case functions.
- Need stop and raise any conflict where current code is better than SPEC.md or where I disagree with the spec.

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
