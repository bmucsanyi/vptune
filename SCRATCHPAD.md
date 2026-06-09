# Scratchpad
Branch: codex-implement-spec @ 2b686bc + P1 substrate extraction in tree

<!-- writing-rules: justified -->
writing-rules justification: filler-adjectives - the word appears only inside the verbatim title of git commit 68a7c05, quoted to identify that commit; rewording would misname the commit.

## TODO
- [x] Phase 0 baseline: local 889 passed / 11 skipped / 82% / ~35s at 2898688; GPU job 399306 submitted on cluster clone (vptune-verify @ 2898688, torch 2.12 overlay) - poll later.
- [x] Pass: public compile entry point (commit d2ab8af).
- [x] Pass: error domains CompileSetupError + RecordValidationError, 7 test assertions tightened (commit 2b686bc).
- [x] P1: extract runtime_values.py substrate leaf (275 names, 218 publicized + docstrings); runtime.py 18,855 -> 15,979 lines; suite green.
- [ ] Commit P1, push to codex-implement-spec-v2.
- [ ] P2: move metric cluster -> vptune/metrics.py (multiply/inverse/sqrt/inner, KFAC/EKFAC/low-rank/GGN-derived/dense/diag/block/matrix-free, damping, preconditioners, *_BY_KIND tables).
- [ ] P3: fisher + ggn clusters -> own modules.
- [ ] P4: vectorization cluster.
- [ ] P5: composition cluster.
- [ ] P6: layout/dtype cluster.
- [ ] P7: compile mechanics cluster (parsers/wrappers; boundary prep stays in runtime).
- [ ] P8: runtime.py residue audit: orchestration + teacher/loss-scaling/microbatch/reference stay; re-check error-domain wiring per module (RuntimeValueError for runtime_values raises, OperatorExecutionError where owned).
- [ ] After conversion: acceptance-bullet fixes from the 79-bullet trace (grades summary below), real-model tests, meta-test strengthening, VERIFICATION.md (untracked).
- [ ] Re-run GPU gate on final tree; certification report.

## Open questions for the user
- SPEC Package Layout amendment for the owner modules (see BLOCKERS.md).

## Uncertain / ideas to explore
- Net +1,045 lines from P1 are docstrings on the publicized substrate API (D103/DOC201 require them). Raise in report.
- Bullet-2248 production gap candidate: runtime-level positive-damping check only on conjugate_gradient (_require_positive_matrix_free_damping); construction-time check at public.py:6187. Verify and fix in bullet-fix phase.
- GPU job list is hand-enumerated in the old hw scripts; full-suite GPU run chosen instead (subsumes the 12).

## Notes to self
- Mover method that works: AST top-level spans + closure check + shadowing scan (params/locals vs module names) + publicize externally-used names + module-object access (runtime_values.x) for monkeypatch stability + ruff --fix prunes copied imports + docstring generator for D103/DOC201 + pure-python harvest of test references (attr, setattr-string, and quoted-string-through-helper forms). The installed vptune in .venv is STALE for import-based scans: always sys.path.insert(0, "src").
- runtime.py param name `execution` is ubiquitous: module names must not collide with locals (first extraction attempt as execution.py was reverted for this).
- Module-level import cycle rule: domain modules must not evaluate runtime attrs at import time; runtime imports domain modules; domain -> runtime calls only inside function bodies (none needed after substrate extraction so far).
- ext.py and run.py imported 4 moved names; fixed to runtime_values imports.
- 79-bullet trace grades: genuine 36, partial 36, name-only 7 (2214, 2248, 2255, 2258, 2282 plus two whose mapped tests assert other behavior). Fix-by-remap candidates: 2225, 2232, 2244, 2253, 2263-2264, 2267 (exhaustive strategy unmapped), 2277, 2281, 2284, 2285. Full agent reports in session transcript.
- Cluster sizes in runtime.py pre-P2: metric ~195 defs / 4.2k lines; fisher 65/1.2k; ggn 39/1k; compile 52/0.9k; composition 32/0.9k; vectorization 38/0.6k; layout 42/0.6k.
- GPU: cluster clone /weka/hennig/owl569/repos/vptune-verify @ 2898688 detached; torch overlay .torch-overlay (2.12.0+cu130); hw_gate.sbatch = full suite on 2x h100; job id 399306 pending. unlearning env: python 3.12.12, numpy 2.2.6, autobatch, transformers 5.5.0.
- origin/codex-implement-spec carries stray commit 68a7c05 (title quoted above; the +1948/-740 rejected work); force-push denied by permission layer; my work pushes to codex-implement-spec-v2. User decision pending on the stray commit.
- Commit messages: regular, no co-author lines.
