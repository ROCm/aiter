# Final Standards review

Pinned review: `git diff 882ea79dd7cc7091f859a5bedac914523ac77bbd...5b09efe5f2843edd0ce104531c55f2fe9e41e3e5`.

Commit inventory: `/root/workspace/tmp_dev/mxfp4-port-notes/final-review-commits.txt`.
File inventory: `/root/workspace/tmp_dev/mxfp4-port-notes/final-review-files.txt`.

This review is the Standards axis only. Production code, introduced tests, acceptance/control/report scripts and relevant user documentation were inspected statically; existing GPU evidence was not rerun. Numerical records and compressed console logs were treated as retained evidence, not as tuning configuration or source requiring a style review.

## Standards sources and applicability

- `/root/.codex/skills/karpathy-guidelines/SKILL.md`: simplicity, surgical changes, existing style, behavioral verification.
- `AGENTS.md`, `CONTEXT.md`, `docs/agents/domain.md`: single context, approved vocabulary and ADRs.
- `docs/adr/0001-mxfp4-search-modes-on-main.md`: prune/full retention, explicit staged strategy on the pruned candidate universe. No contradictory new public mode or candidate registry was found in this axis.
- `CONTRIBUTE.md`, discovered through `README.md:161`: Code Style Guidelines at lines123–144 and DCO at560–572. This is present on the fixed base, not introduced by this port.
- `pyproject.toml`, `.github/workflows/pre-checks.yaml`, `.githooks/pre-commit`: Black/Ruff style rules are tool-enforced and omitted from findings. No annotation or DCO checker is configured.
- `.github/instructions/aiter-ops-triton.instructions.md` applies only to Triton paths; `aiter/ops/triton/configs/CLAUDE.md` is likewise outside this diff. Their tuned JSON and test configuration rules do not govern FlyDSL acceptance artifacts.
- Approved spec Testing Decisions explicitly endorses existing internal fake measurements, call counts and fake-context/event supervision. Those repo/task-specific choices override generic objections to seam-driven tests.

## Findings

### S1 — Documented rule: new signatures omit type hints (P2)

`CONTRIBUTE.md:127` says **“Use type hints for function signatures.”** The pinned diff adds308 completely unannotated function/method signatures. Production examples are:

- `aiter/utility/base_tuner.py:356` `_normalize_tuned_csv(self, tunedf)`;
- `aiter/utility/base_tuner.py:359` `_normalize_tuned_identity(self, tunedf)`;
- `aiter/utility/base_tuner.py:580` `_summary_shape_keys(self)`;
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:5974` `_already_tuned_mask(self, dedup_cols)`;
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:6497–7106` introduces15 further completely unannotated configuration, runtime, normalization and plan helpers.

The same rule is missed in new acceptance functions, for example `scripts/run_gpu_task.py:30` `write_json`, `scripts/configured_replay.py:31` `write_json`, `04/summarize_calibration.py:15` `finite_number`, and all new test functions/fakes.

A complete, reproducible AST inventory is retained at `/root/workspace/tmp_dev/mxfp4-port-notes/final-standards-typehint-inventory.json` (309 entries). It only considers function definition lines added in zero-context diff hunks; it does not flag untouched legacy signatures.308 are wholly untyped. The remaining entry, `aiter/aot/flydsl/mxfp4_moe.py:105` `parse_csv`, has parameter annotations but no return annotation; do not count this as a hard breach of the literal rule merely because it lacks a return hint.

Repair recommendation: annotate introduced signatures, including test fixtures and helpers, with useful lightweight types. Do not annotate every unchanged legacy function or introduce runtime GPU imports solely for type hints. A source AST check can verify that all wholly untyped introduced signatures have been addressed; use existing behavioral tests after a types-only patch.

### S2 — possible Duplicated Code (judgement, P3)

`docs/mxfp4_staged_main_port/scripts/default_replay.py:240–272` and `configured_replay.py:404–436` repeat the same synchronized warmup/profile/trace aggregation, including:

```python
kernels = [event for event in events
           if event.get("cat") == "kernel" and event.get("ph") == "X"]
kernel_sum = sum(event["dur"] for event in kernels) / args.iters
span = (max(event["ts"] + event["dur"] for event in kernels)
        - min(event["ts"] for event in kernels)) / args.iters
metric = float(get_trace_perf(prof, args.iters))
```

The identical tensor fingerprint expression also appears at `default_replay.py:36`, `configured_replay.py:45`, and `04/alternating_replay.py:137`. Every measurement correction currently needs to be applied to multiple new executable scripts. Extract the shared profile aggregation/fingerprint shape into a small acceptance helper, preserving the individual scripts’ differing fixture, configuration and result contracts. Do not collapse all replay behavior into a flexible generic framework. The source scripts are executable reproduction tools, while the task files/commits preserve the actual historical runs; shared helper extraction need not rewrite raw evidence.

### S3 — possible Data Clumps (judgement, P3)

`docs/mxfp4_staged_main_port/03/create_fixtures.py:12–13` declares a positional11-field request format and `:345–347` unpacks it:

```python
# (id, model, token, BM, BN, BK, epilog, FP8, FP4, limit, clamp fixture)
fixture_id, model, token, bm, bn, bk, epilog, fp8, fp4, limit, clamp = request
```

A small frozen named request type or `NamedTuple` would make the repeated adjacent flags/tile values legible and avoid swapping fields while extending specified-pair data. This is a heuristic recommendation, not a documented hard rule or a correctness finding. Keep the existing admitted-pair selection and fixture result schema.

## DCO convention observation (not an additional hard finding)

All82 commits in the reviewed range lack `Signed-off-by` trailers (including both implementation and evidence commits). `CONTRIBUTE.md:565` demonstrates `git commit -s`, and`:568` explains its trailer. However, the text does not explicitly say every commit must contain a trailer, and no local/CI enforcement was found. It should be recorded as the documented contribution convention not used, rather than manufacturing a requirement to rewrite all82 commits or invalidate recorded implementation SHAs. A signed final repair/adoption commit can document the final delivered contribution while retaining the evidence graph; any later publication/DCO requirement would need to be assessed in that publication workflow. This task explicitly requests only a local integration branch.

## Suppressed smell candidates and clean findings

- Stage candidates, pipeline pair identities, screening keys and accumulation mode are named domain concepts from CONTEXT/spec; the small frozen types are justified and not speculative generality.
- `_normalize_tuned_csv`, `_summary_shape_keys` and `_already_tuned_mask` are actually overridden by the MXFP4 tuner and prevent ordinary-tuner semantic changes; the base methods are purposeful seams, not Middle Man hooks added for hypothetical callers.
- Dedicated staged process supervision and best-effort cleanup own Pipes/processes/GPU slots. They are explicitly required; catch-all cleanup does not imply a generic process-framework refactor.
- Existing kernel dimension/sorting abbreviations (`BM`, `BN`, `BK`, `sti`, `sw`, `sei`, `nvi`, `h`, `e`, `ne`) match adjacent tuner and wrapper conventions and describe established domain fields. They are not flagged as Mysterious Names.
- G1 and G2 planning projection comprehensions are similar but produce separate stage-specific types. A generic abstraction solely to hide a few lines would conflict with simplicity guidance, so no additional duplicate-projection finding.
- Replay dictionaries and CSV shape keys preserve the explicitly required public schema; do not change them into sidecars, new cache keys or automatic configuration associations under a Primitive Obsession label.
- Current docs distinguish stage screening score, pipeline-pair validation latency, candidate universe and tuning wall time. Avoided glossary synonyms were not introduced as positive replacements. Negative explanations of “GPU lock” do not replace the approved per-GPU measurement exclusivity term.
- No unrelated production kernel backend change or new third-party dependency. The existing tuner lives under a CK-named directory but the port concerns FlyDSL dispatch/planning; no PR backend split is relevant to the authorized local integration delivery.

## Totals

One grouped documented-standard finding (type hints), two labelled heuristic findings (Duplicated Code, Data Clumps). DCO is recorded as a convention observation, not a fourth blocking finding. No GPU execution was performed by this review.
