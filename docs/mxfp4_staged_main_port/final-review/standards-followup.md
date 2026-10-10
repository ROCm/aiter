# Final Standards fix verification

Pinned integration: `ff4e6db25cb98bab1959fffc67d75526fd017e2c`.
Repair diff: `git diff 5b09efe5f2843edd0ce104531c55f2fe9e41e3e5...ff4e6db25cb98bab1959fffc67d75526fd017e2c`.
Fixed original base: `882ea79dd7cc7091f859a5bedac914523ac77bbd`.
This follow-up checks original S1/S2/S3 only; no GPU work or source changes were performed.

## S1 — type hints: presence fixed, inaccurate new hints require correction

An independent static AST/diff inventory across fixed base to new head found340 introduced/modified function definition lines and zero wholly untyped signatures. All originally inventoried signatures are now annotated or supplied by the shared fingerprint helper. The original omission finding is therefore addressed at the presence level. Direct annotations are useful in production normalization/device helpers; string/TYPE_CHECKING annotations avoid adding GPU imports merely to express types.

However, the repair introduces definite incorrect annotations. These are in the repair scope and should be corrected before claiming S1 fully resolved:

1. `docs/mxfp4_staged_main_port/04/summarize_calibration.py:93–95` `rank_retention(..., console: Path)` and`:141` `staged_fallback_groups(console: Path)` both pass `console` directly to `re.findall`. The actual caller passes text read from a log at`:253`, and the driver likewise passes its read text. Both require `console: str`, not a path. At`:54`, `comparison` also accepts missing candidate records, so its record parameters should reflect `dict[str, object] | None`, not a required candidate dictionary.
2. `docs/mxfp4_staged_main_port/scripts/default_replay.py:109` and `configured_replay.py:204`: `capture_g1` declares a tuple of tensors but delegates to direct `flydsl_mxfp4_gemm1`, which fills supplied buffers and has no return statement (`aiter/ops/flydsl/mxfp4_gemm1_kernels.py:187` onward). Return hint should be `None`.
3. `configured_replay.py:173–180` `capture_reduction` and`:194` `capture_quantized_scatter` declare tensor returns but delegate to out-buffer operations returningNone (`aiter/ops/flydsl/moe_kernels.py:1019`, `aiter/ops/moe_mxfp4_aux.py:170`). Return hints should be `None`.
4. `op_tests/tuning_tests/test_mxfp4_staged_search.py:84–86`: `screen_topk: int` is parameterized with strings`"0"`, `"-1"`, `"2.5"`, passed unchanged to the string CLI helper. Hint should be`str`.
5. `op_tests/tuning_tests/test_mxfp4_configured_paths.py:303–315`: limit/fp8/fp4 values are environment strings orNone, including empty strings and`"5"`/`"1"`; hints`float | None`/`bool`/`bool` are wrong. Use`str | None` for all three.
6. `op_tests/tuning_tests/test_mxfp4_tuner_lifecycle.py:167–182`: `controlled_mxfp4(... attempts: list[int])` appends`dict(row)`. Hint should be`list[dict[str, object]]`.
7. `op_tests/tuning_tests/test_mxfp4_staged_supervision.py:343–347`: cancellation `failure: Exception | str` is parameterized with `KeyboardInterrupt`, which inherits directly fromBaseException. Use`BaseException` for this specific parameter; no string is supplied there.
8. `test_mxfp4_staged_supervision.py:410–414` and`:433` schedule helpers label`batch` as`list[dict[str, object]]` but immediately unpack each batch item as`(_, row, _, _)`; the actual batch is scheduler payload tuples. The parameter should name its tuple shape. This is distinct from the returned list of row dictionaries.

These annotations do not change runtime behavior and do not require repeat GPU calibration. The documented standard asks for type hints; inaccurate hints make the repair misleading even though an annotation-presence check passes. Broad `object` use for fake arbitrary values is not separately flagged, and no demand for a repository-wide type checker is introduced.

## S2 — possible Duplicated Code: resolved

`replay_common.py` owns the same raw CPU tensor byte fingerprint expression and trace kernel filter/sum/span operations used in the original scripts. Default/configured replay both call`trace_kernel_metrics`; all three tools use shared`fingerprint`. Warmup loops, per-call synchronization, profiler activities, main`get_trace_perf` timing aggregation and differing fixture/output contracts remain unchanged. The extraction uses two small functions and adds no generic runtime framework. Remaining similar wrapper capture loops concern different effective configuration checks and are not an outstanding S2 repair requirement.

The two helper tests assert independent literal expected metrics and a known literal SHA256, rather than merely comparing the helper to itself. Existing294-check evidence and two root-coordinated exact run-only task results are retained. Reviewer inspected records: both exit0, finite outputs, cosine errors0.011174413919489035 and0.011281507664015256; physical UUIDGPU-5d8a72a332aa124c and PCI0000:f5:00.0; run-only environment`1`. These records support the extraction and were not rerun by this reviewer.

## S3 — possible Data Clumps: resolved

`FixtureRequest(NamedTuple)` gives all11 request fields names and the execution code reads named fields. Independent AST literal comparison confirms all22 request values and order equal the pre-fix data. Retained empty-visibility regeneration confirms admitted fixture metadata/configuration match by stable id while measured fields remain original evidence. No new configuration schema or experiment is introduced.

## Runtime and submission convention

Independent annotation/import-stripped AST comparison confirms the three changed production files (`base_tuner.py`, `mxfp4_moe.py`, `gemm_moe_tune.py`) retain identical executable AST. The repair diff adds type imports plus the deliberate replay helper/request representation changes described above.

Direct repair commits and final merge carrySigned-off-by; original82 reviewed commits remain unchanged. One internal synchronization merge lacks a trailer, but the prior DCO observation did not establish a mandatory per-commit trailer rule, so it remains non-blocking and does not justify rewriting evidence SHAs.

## Verdict

S2resolved; S3resolved; S1annotation omission resolved but correction of clearly inaccurate new hints remains. One grouped follow-up issue within S1; no new runtime correctness finding. Final Standards clearance should follow this types-only correction.
