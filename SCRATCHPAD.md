# Scratchpad
Branch: codex-implement-spec-v2 local @ 0046714 = origin

<!-- writing-rules: justified -->
writing-rules justification: filler-adjectives - the word appears only inside the verbatim title of git commit 68a7c05, quoted to identify that commit; rewording would misname the commit.

## TODO
- [x] Narrow plan items 1-5; ten-pass runtime conversion; five-layer package restructure with layering meta-test; SPEC layout amendment.
- [x] Acceptance work: mapping extensions (14 bullets); new behavioral tests (2248 four-solve-path admission, 2250 round-trips + block sqrt, 2255 cohort pinning, 2258 replay identities, 2262 fullgraph graph-break, 2266 exhaustive); eigenbasis sqrt spec divergence FIXED (symmetric + factored apply); seven weak mapped tests strengthened (match pins / real assertions).
- [x] Real-model tests: transformer + conv through public API; CPU flash-SDPA double-backward/forward-AD constraints pinned to math kernel.
- [x] Meta-tests strengthened: behavioral-assertion requirement for every mapped test; manifest literal coverage restricted to executable test code (decorators + helpers + classes count; module-level dicts do not). All three meta-tests green.
- [x] Error domains: CompileSetupError, RecordValidationError, RuntimeValueError (145 raises in runtime_values) wired, exported, docstrings updated.
- [x] VERIFICATION.md written (untracked): genuine 57, partial 21, name-only 1 (2214).
- [x] GPU gates: 399306 (2898688) 899/1; 399380 (9bc8204) 899/1; 399381 (704c04b) 900/1 - all COMPLETED 0:0 on 2x h100 full suite.
- [ ] Poll 399385 (final tree 0046714), then deliver certification report.
- [ ] Continuation worklist (next session): remaining partial bullets per VERIFICATION.md tail (2214 explicit contradictions, 2230 fused higher-order, 2236 PSD enumerations, 2239 gating, 2241 chunk, 2251 inner representations, 2254 Lanczos tol, smaller items).

## Open questions for the user
- None pending.

## Notes to self
- Local: 907 passed, 11 skipped, ~36s, coverage ~83%. Meta-tests now: name-existence + literal-in-executable-code + behavioral-assertion + layering.
- GPU: ssh -i ~/.ssh/mlcloud_auth -p 2221 owl569@134.2.168.205; /weka/hennig/owl569/repos/vptune-verify; .torch-overlay torch 2.12.0+cu130; hw_gate.sbatch full suite on 2x h100.
- Error taxonomy state: VPTuneError > Admission/ReferenceFailed/RecordFormat/NoPassedCandidate/StaleRecord/Measurement/Materialization > {CompileSetup, RecordValidation, RuntimeValue}. Operator-execution split left as the one open taxonomy judgment (engine execution raises remain materialization-domain).
- Commit style: regular messages, no co-author; make lint-fix before commit (pre-commit hook).
