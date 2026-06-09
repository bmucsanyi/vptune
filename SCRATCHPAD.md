# Scratchpad
Branch: codex-implement-spec local @ 704c04b = origin/codex-implement-spec-v2

<!-- writing-rules: justified -->
writing-rules justification: filler-adjectives - the word appears only inside the verbatim title of git commit 68a7c05, quoted to identify that commit; rewording would misname the commit.

## TODO
- [x] Baseline: local 889/11 at 2898688; GPU 399306 full suite at 2898688: 899 passed, 1 skipped, 0:0.
- [x] Narrow plan items 1-5: compile API (d2ab8af), error domains slice (2b686bc), runtime conversion in 10 passes (89daee1..9bc8204), runtime.py 18,855 -> 2,705.
- [x] GPU 399380 full suite at 9bc8204: 899 passed, 1 skipped, 0:0.
- [x] Five-layer package restructure (704c04b): core/ axes/ engine/ tuning/ adapters/ + root surface; all imports rewritten (42 files); schemas+io placed in tuning per measured import graph; layering meta-test added (test_core.py test_package_layers_have_one_directional_imports); SPEC Package Layout amended with the tree + layer rule; BLOCKERS back to None. Local: 890 passed, 11 skipped.
- [x] GPU 399381 submitted at 704c04b - POLL NEXT.
- [ ] Poll 399381 (expect 900 passed, 1 skipped on GPU).
- [ ] Acceptance-bullet fixes (remap-first; grades below), real-model tests, meta-test strengthening, VERIFICATION.md (untracked).
- [ ] RuntimeValueError wiring for engine/runtime_values raises; OperatorExecutionError placement.
- [ ] Certification report per DoD shape on the final tree.

## Open questions for the user
- None outstanding (layout amendment authorized and done; branch decision settled).

## Uncertain / ideas to explore
- P11 lesson: run_with_backend_settings/dot_runtime/tree ops call layout/anchors functions, so they are orchestration, not leaf; one-directionality is now enforced at the PACKAGE level instead (engine-internal module cycles are call-time-safe and intra-layer).
- Bullet-2248 enforcement gap candidate now in engine/metrics.py.

## Notes to self
- Module paths after restructure: vptune.core.{data,identities,operators,tensor_tree}; vptune.axes.{candidates,admission}; vptune.engine.{runtime,runtime_values,derivatives,ggn,fisher,metrics,vectorization,composition,layout,memory,compile,attention,anchors,checks,reference}; vptune.tuning.{run,measure,select,selection_core,cohorts,autobatch_bridge,schemas,io}; root: errors, public, ext.
- Tests import aliases unchanged (runtime_module = vptune.engine.runtime etc.); `from vptune.core import operators as ops` re-added in 5 test files + public.py after the rewriter dropped asnames (ruff then pruned them).
- GPU: ssh -i ~/.ssh/mlcloud_auth -p 2221 owl569@134.2.168.205; /weka/hennig/owl569/repos/vptune-verify; .torch-overlay torch 2.12.0+cu130; hw_gate.sbatch full suite on 2x h100. Jobs: 399306 (2898688) 899/1 PASS; 399380 (9bc8204) 899/1 PASS; 399381 (704c04b) pending.
- 79-bullet trace: genuine 36, partial 36, name-only 7 (2214, 2248, 2255, 2258, 2282 + 2). Remap-first: 2225, 2232, 2244, 2253, 2263-2264, 2267 exhaustive, 2277, 2281, 2284, 2285.
- origin/codex-implement-spec still carries 68a7c05; user said codex-implement-spec-v2 is the branch.
- Commit messages: regular, no co-author lines. VERIFICATION.md untracked. CODEX_SESSION_ANALYSIS.md stays untracked (excluded from commits).
