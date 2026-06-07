# Scratchpad
Branch: codex-implement-spec @ b112e52 + uncommitted validation audit fixes

## TODO
- [x] Read SPEC.md in full.
- [x] Read FEATURES.md in full.
- [x] Read FAILURE_MODE_ANALYSIS.md in full.
- [x] Read REPO_AUDIT.md and extract mistakes to avoid.
- [x] Build a source-backed engineering checklist from current primary sources.
- [x] Patch package metadata, security files, dependency bounds, and sdist contents.
- [x] Patch candidate generation to reject empty extension-axis value lists.
- [x] Patch fresh-install PyTorch NumPy warning by adding explicit NumPy runtime dependency.
- [x] Patch timing policy validation.
- [x] Patch public target and policy identity validation.
- [x] Patch selection/search policy integer validation.
- [x] Run `make lint-fix`.
- [x] Run `make lint`.
- [x] Run full local `make test`: 889 passed, 11 skipped.
- [x] Run `git diff --check`.
- [x] Build current wheel/sdist at `/private/tmp/vptune-audit-final-dist-5`.
- [x] Run `twine check` on current wheel/sdist.
- [x] Fresh-install current wheel/sdist into Python 3.14 envs.
- [x] Run warning-as-error imports from both fresh installs.
- [x] Run warning-as-error README-style CPU tuning workflow from both fresh installs.
- [x] Run `uv pip check` in both fresh installs.
- [x] Run `pip-audit` on final wheel env by path.
- [x] Cancel stale Ferranti job `398531`.
- [ ] Commit and push final validation audit fixes.
- [ ] Pull final commit on Ferranti.
- [ ] Submit current-code hardware tests to `h100-ferranti`.
- [ ] Poll current-code Ferranti job and read output.

## Open questions for the user

## Uncertain / ideas to explore
- Direct git dependency on `autobatch` works for local, wheel, and sdist installs, but public advisory tooling skips it because it is not a PyPI package.
- `pyproject.toml` declares `import-names = ["vptune"]`, but emitted wheel metadata stays at Core Metadata 2.4 because current Twine rejects Core Metadata 2.5.
- Remaining pytest warning filters are exact PyTorch internal warning filters for `torch.jit` and `torch.fx` paths required by the test suite. Fresh install imports and README workflow pass with `-W error`.

## Notes to self
- `BLOCKERS.md` currently contains `None`.
- `make test` final local count after validation fixes: 889 passed, 11 skipped, coverage 82%.
- Current wheel metadata includes SPDX license, license file, NumPy, PyTorch `>=2.12,<2.13`, and no legacy license classifier.
- Current sdist includes `.github/CODEOWNERS`, `.github/dependabot.yml`, `SECURITY.md`, SPEC/FEATURES/BLOCKERS, tests, source, README, LICENSE, Makefile, pyproject, and uv.lock.
- `settings_product` now raises `AdmissionError` for an empty axis value list instead of leaking an unbound local.
- `TimingPolicy` now rejects invalid thresholds, negative warmups, and nonpositive measured-call counts.
- Public `Target` now rejects empty/duplicate devices, empty accelerator, duplicate allowed values, non-tuple allowed values, and empty allowed entries.
- `DeterminismPolicy` and `EnvironmentPolicy` now require string keys and JSON-compatible values.
- `SelectionPolicy` now rejects invalid `near_fastest_multiplier` and non-integer or nonpositive `compile_call_horizon`.
- `SearchPolicy` now rejects bool/non-integer retained counts, compile horizons, and variance repeat counts.
- Stale Ferranti job `398531` was canceled because it targeted commit `b112e52` before the validation fixes.
