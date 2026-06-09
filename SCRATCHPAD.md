# Scratchpad
Branch: codex-implement-spec local @ 9bc8204 = origin/codex-implement-spec-v2

<!-- writing-rules: justified -->
writing-rules justification: filler-adjectives - the word appears only inside the verbatim title of git commit 68a7c05, quoted to identify that commit; rewording would misname the commit.

## TODO
- [x] Baseline: local 889/11/82% at 2898688; GPU job 399306 on 2x h100 at 2898688: FULL SUITE 899 passed, 1 skipped (CPU-only test), exit 0:0.
- [x] Narrow plan item 1: public compile entry (d2ab8af); zero cross-module underscore access repo-wide after conversion.
- [x] Item 2 (first slice): CompileSetupError + RecordValidationError domains (2b686bc).
- [x] Items 3-5: runtime.py conversion in 10 passes, all green: runtime_values (89daee1), metrics (22eb522), ggn (36f2653), fisher (acb61a0), vectorization (54c8d43), composition (6543726), derivatives (c164b81), layout (70bec88), memory (9ffa1d7), compile (9bc8204). runtime.py 18,855 -> 2,705 lines (orchestration: factories, runners, microbatch, teacher, loss scaling, reference checks).
- [x] P11 attempt (push shared helpers into runtime_values) REVERTED: several helpers call domain functions (layout foreach, anchors), so they are orchestration, not leaf; pushing them down inverts layering and breaks import-time annotations. Bidirectional module-object imports are call-time-safe and suite-proven; composition and compile re-enter orchestration by design (wrapper layers).
- [x] Final-tree GPU gate submitted: job 399380 at 9bc8204 - POLL NEXT.
- [ ] Poll job 399380; expect 899 passed, 1 skipped.
- [ ] Acceptance-bullet fixes (remap-first) from the 79-bullet trace; real-model tests (torch-native transformer + conv net); meta-test strengthening; VERIFICATION.md (untracked).
- [ ] RuntimeValueError wiring for runtime_values raises (now lexical); OperatorExecutionError where owned.
- [ ] Certification report per DoD shape on the final tree.

## Open questions for the user
- SPEC Package Layout amendment for the new owner modules (BLOCKERS.md entry).
- origin/codex-implement-spec carries stray commit 68a7c05 (+1948/-740 rejected work); force-push denied by permission layer; all my work is on codex-implement-spec-v2. Disposition is your call.

## Uncertain / ideas to explore
- Net source delta of the conversion: src 45.8k vs 49.0k pre-refactor IS NEGATIVE?? recount at report time (docstrings added but import dedup and ruff fixes removed code).
- Bullet-2248 enforcement gap now lives in metrics.py; verify during bullet fixes.

## Notes to self
- Mover machinery proven over 10 passes; per-pass gotchas list kept from previous scratchpad (zsh word-split, parenthesized imports, stray docstring from import-block copy, F823 publicize collisions, string-form patch targets, stale .venv for import scans).
- GPU: ssh -i ~/.ssh/mlcloud_auth -p 2221 owl569@134.2.168.205; /weka/hennig/owl569/repos/vptune-verify; .torch-overlay torch 2.12.0+cu130; hw_gate.sbatch = full suite on 2x h100; jobs: 399306 baseline COMPLETED 899/1; 399380 final-tree pending.
- 79-bullet trace: genuine 36, partial 36, name-only 7 (2214, 2248, 2255, 2258, 2282 + 2). Remap-first: 2225, 2232, 2244, 2253, 2263-2264, 2267 exhaustive, 2277, 2281, 2284, 2285.
- Commit messages: regular, no co-author lines. VERIFICATION.md untracked.
