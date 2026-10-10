# MXFP4 staged port acceptance

Status: ready-for-agent

The [spec.md](spec.md) and the approved [four tickets](issues/README.md) incorporate the authorized pre-implementation reviewer corrections and are ready for implementation. No implementation or acceptance execution has started; the [work map](map.md) tracks ticket state. This reference supplies fixture rationale and measurement details. Historical experiment evidence does not replace integrated-port validation.

## Functional acceptance

### Policy and lifecycle checks

Use shape-level interfaces and fake measurement outcomes to test the actual search decisions. Cover the shared main enumeration and its `full_search` parameter, exact stage projection, fixed/derived contract checks, original-order ties, screening top-K 1/2/greater-than-available/invalid values, atomic versus non-atomic grouping, G1 ranking reuse, and full/prune isolation from the staged supervisor.

Include M_est `{3,4,15,16,32,33,63,64,65,128,129}` with main's pruning rules. Demonstrate that staged fallback visits returned-screening-failed stage candidates still present in the pruned universe, exhausts remaining pairs when required, and cannot restore configurations excluded by M_est pruning. Validate every returned result only once; bootstrap results do not compete or enter that cache.

Test no usable reference, no stage-screening success, empty enumeration, invalid pairs, non-finite time/error, catchable timeout/failure, no winner and fatal worker death. Missing validated references or an empty stage shortlist must use direct exhaustive normal pair validation in the same admitted pruned group, reusing returned pair outcomes. Assert expansion/shape-failure behavior and later-shape continuation. Deterministic fixture failures must not lose a valid prune pair, including atomic reference failure with a healthy non-atomic pair.

Supervisor tests coordinate process events directly. Exercise result-before-exit, exit-before-read, EOF with no result, fatal exit, delayed exit, bounded active Process handles, cancellation and unexpected IPC failures; prove same-logical-GPU measurement exclusivity and release only after worker exit. Simulate unsupported/mixed GPU properties and visible logical-device mapping. GPU execution itself remains single-card.

Runtime-device preflight applies to MXFP4 prune/full as well as staged. Use controlled outcomes and simulated properties to verify refusal before work when an actual participating device is unsupported, acceptance when an unsupported visible device is unused, and result gfx/CU derived from the assigned supported runtime device despite differing build/global identity. Retain each mode's parallelism selection and supervision; do not apply the MXFP4 gfx950 restriction to other tuners. Ticket 01 uses these fixtures without real GPU measurements; ticket 02 is the first real GPU integration and applies the physical-identity/idle/process discipline below.

CSV tests exercise preprocessing through final summary and exit status, not just `post_process`. Cover all/non-all, same/separate-file, empty runs, supported/unsupported gate metadata, an unsupported row already matching a tuned shape, existing valid rows, tagged fallback rows, mixed architectures, numeric defaults and ordinary-tuner compatibility. Failures remain visible in summary and yield nonzero final status while valid later winners persist.

Main already exits nonzero for failed or unfinished rows; verify correct stable-key accounting rather than introduce a new exit policy. Run at least one real non-MXFP4 tuner, such as ordinary fMoE or GEMM, through its own input/result/CSV-write/sort/summary path with controlled measurement outcomes. Cover successful writes, ordering/deduplication, failure and unfinished rows, mixed partial success, empty runs and zero/nonzero command exit status. Preserve that tuner's schema, numeric/NT fields, timing/profile semantics and support contracts. A toy base-class fixture alone is insufficient; this infrastructure regression does not require GPU kernel tuning.

Pure policy helpers should be testable without a GPU kernel. Full-module integration may require the repository's ROCm/FlyDSL imports; label that dependency accurately and do not present GPU-visible collection as a GPU-free environment.

### Execution parameters, compilation and replay

Check parameter forwarding at direct G1 launch, production wrappers, reference bootstrap and end-to-end validation. FP8 reduce must apply route weights exactly once, including non-uniform weights; custom limit must actually clamp deliberately chosen inputs and match reference behavior. FP4 scatter must use packed payload/scales and quantized scatter.

Capture the effective limit/flags once per invocation and keep AITER_MXFP4_TUNE_SWIGLU_LIMIT, AITER_FLYDSL_STAGE2_FP8 and AITER_MXFP4_INTERMEDIATE unchanged in the parent and workers until the tuning run ends. Test spawned-worker inherited effective settings against the snapshot for unset/empty defaults, each enabled flag, their combination and custom limit. The production wrapper still reads the flags on every call; verify the actual reduce/scatter path agrees with the recorded snapshot. Set matching environments before separate AOT/replay invocations. The accepted contract prohibits in-run environment mutation and does not require a new runtime mutation detector or CSV-binding check.

Verify both stage2 flags together: reduce selects its FP8 path, scatter its FP4 path, and atomic retains its established output handling. Preserve the common non-atomic screening context while retaining full stage execution identities. Never combine scores or validation results from different effective settings.

AOT job generation and invocation must match production's effective dtype, persistence, scale buffers and normalized activation parameters. Add an explicit `--swiglu-limit` to the MXFP4 AOT entry point and keep omitted-argument callers compatible; document a reproducible way to supply the tuning limit to batch AOT. Production continues to receive its explicit API parameter. Final CSV columns do not carry this extra deployment metadata.

Winner-file/configuration association is the caller's responsibility. For these tests and calibration, select separate outputs for the recorded settings and pass the matching limit/flags to tuner, AOT and replay. No test should require a new sidecar, automatic output/environment binding or mismatch-rejection feature; those were explicitly left outside this port by the user. Existing pipeline parameter forwarding, numerical behavior and matching precompiled entries still require verification.

Usage documentation must show the existing-row hazard: when a CSV already contains a default-limit result at the same shape/device key, a limit 5 invocation without `--all` can silently skip it; with `--all`, a valid new winner replaces that row. Different output flags collide in the same way. The caller selects the intended configuration's file; `--all` forces retuning and does not separate configurations. Calibration also uses per-mode outputs or deliberate retune selection so that prune, staged and full each execute their prescribed search.

Use isolated configuration files and controlled caches to prove production dispatches the actual selected GEMM1/GEMM2 names. Replay must be numerically valid and use matching precompiled entries; an `OK` produced by heuristic/fallback kernels or a different runtime compile key does not satisfy this requirement.

Also verify the effective input quant dtype and actual prequantization path, not just names. Small default Swiglu or Situv2 calls may resolve A16W4; an A4W4 test must set and confirm its intended FP4 input contract and the corresponding config lookup/metadata. Match sort backend/padding, BM input quantization, G1 intermediate layout and complete G2 output handling between screening fixtures, pair validation and production. See the [pipeline grouping audit](pipeline-grouping-audit.md).

## Numerical acceptance (confirmed Q12)

Keep the main MXFP4 `--errRatio` default of 0.1 with finite error required. The existing metric is `1 - 2 * dot(ref, out) / (norm(ref)^2 + norm(out)^2)`; it is not a relative-error percentage. Use the same Torch reference for every environment, including FP4 scatter, and report measured error per path. A lossy pair exceeding the configured gate is invalid; do not silently relax the gate or make the reference copy the extra quantization to conceal it.

## GPU fixtures and matrix (confirmed Q13)

Select and validate the intended physical GPU before mapping it to a logical device; bare `HIP_VISIBLE_DEVICES=0` is not a physical-card selector. Keep separate output CSVs and artifacts per effective configuration, with identical input/routes and measurement settings within each comparison.

The user subsequently selected physical GPU7 for the crossed-pair experiment. Use a fixed physical card for each comparison, check it is idle before every independent-process task, run those tasks sequentially, and confirm PCI/UUID after visible-device remapping. On this host, monitoring-tool GPU7 is selected by `ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c` with `HIP_VISIBLE_DEVICES=0`; bare HIP ordinal7 identifies another card. The logical0 notation alone is insufficient to establish which physical card was used.

### Specified-pair correctness

- Default MiniMax `(token=128,H=3072,I=384,E=256,routing_topk=8)` for BM16/BM32 and `(2048,3072,384,256,8)` for BM64/BM128. Exercise atomic/reduce and BM128 scatter explicitly in each legal selected range.
- GPT-OSS Swiglu `(128,3072,512,128,4)` and `(2048,3072,512,128,4)` for extended environments. These supply G2 K128/K256 and BM16/32/64/128. Exercise FP8 reduce, FP4 scatter with BN128/BN256, a positive custom limit of 5, and combined stage2 options with/without the custom limit. Direct pair selection here is a test fixture, not another tuner candidate source.
- Retain the small `(4,3072,512,128,4)` winner subprocess smoke and add default Situv2 production replay. Choose an existing supported tuple and verify auxiliary coverage before execution. The small smoke has M_est=1 and cannot establish scatter coverage.

### Default calibration: MiniMax Silu

Fixed `(H=3072,I=384,E=256,routing_topk=8)` from target main's model input CSV:

| Tokens | M_est | First-pass searches |
| --- | --- | --- |
| 64 | 2 | full, prune, staged |
| 128 | 4 | prune, staged |
| 512 | 16 | prune, staged |
| 1024 | 32 | prune, staged |
| 2048 | 64 | full, prune, staged |
| 4096 | 128 | prune, staged |
| 8192 | 256 | full, prune, staged |

This is the agreed 17 shape-mode searches. I384 remains unpadded and legal through the layout-v2 K128 path.

### Extended calibration: GPT-OSS Swiglu

Fixed `(token=2048,H=3072,I=512,E=128,routing_topk=4)`; run prune and staged for each row:

| Configuration | FP8 reduce | FP4 scatter | Swiglu limit |
| --- | --- | --- | --- |
| Default | Off | Off | Default 7 |
| FP8 | On | Off | Default 7 |
| FP4 | Off | On | Default 7 |
| Custom limit | Off | Off | 5 |
| Combined | On | On | 5 |

This adds 10 shape-mode searches, for the agreed 27 first-pass searches. The both-flags/default-limit combination is covered with specified pairs only; it adds no tuning invocation beyond the 27-search matrix. Extended configurations report prune-to-staged effects. Do not claim their full-to-prune or full-to-staged regret without the corresponding full data.

### Search cost before calibration

Before launching the matrix, use the current implementation's shared pair enumeration to record admitted G1/G2 and pair counts by BM/accumulation for each shape/mode, with explicit unpruned counts for MiniMax tokens 64, 2048 and 8192. This counting step launches no GPU kernels or JIT compilation; imported tooling requirements remain accurately labeled. Static enumeration counts legal proposed pairs, not pairs proven numerically valid at runtime. Record staged screening/shortlist work separately and retain exhaustive pruned-pair fallback as its validation upper range.

Before starting each full anchor, record a projected wall-time range and its assumptions in the calibration artifacts. Use representative elapsed per-pair attempt overhead observed during prescribed focused acceptance on the same card/toolchain; include case/reference preparation, compilation/cache state, correctness work, warmup/iterations, profiling and failed attempts. Winner kernel latency multiplied by pair count omits most tuning overhead and is not an estimate. Disclose extrapolation uncertainty, especially for the large-token anchor and unmeasured compilation keys, and update the estimate with observed progress. Reuse required measurements rather than adding a new search or benchmark matrix.

Soft SIGALRM timeouts do not establish a hard total-time bound, and a permanently live hang remains unbounded. Cost planning neither adds a hard deadline nor automatically skips required anchors; incomplete searches remain incomplete acceptance evidence. Report actual wall time and count alongside the estimate when each anchor finishes.

## Performance reporting (confirmed Q14)

Use default screening top-K 2. Report winner latency/regret, validation counts, per-shape tuning wall time and summed wall time over the seven default calibration shapes. Report extended-environment observations independently. This port has no hard latency-regret or tuning-speedup thresholds; the earlier ≤5% regret and 2x speedup proposals were not accepted.

Full-to-prune and full-to-staged regret on the full anchor rows is disclosed separately. Main's existing M_est exclusion cost is not attributed to staged screening. For any missing or invalid reference result, report the comparison unavailable or failed rather than computing a fictitious regret.

All paired prune/staged points remain in the calibration outcome table even when one run has no winner. Count prune-valid/staged-invalid outcomes and classify independent fixture, validation, fatal-worker or incomplete-run causes separately from timing regret. The revised fallback eliminates independent-fixture skip as a reason to forgo normal admitted-pair validation, not the possibility of fatal/incomplete failures.

Separate search cost from replayed winner cost. Record cold/warm compilation conditions and use comparable caches; keep warmup/iteration settings consistent with main. Alternate replay of the two selected pipeline pairs on the same fixed data to verify negative, anomalous or noisy latency differences. Re-search only when a concrete candidate/ranking concern remains. Record median and dispersion. Do not change the agreed screening top-K or fallback policy automatically to improve the report.

When investigating a concrete staged/prune discrepancy, report the prune winner's G1/G2 screening ranks and shortlist retention. A small crossed pair set and per-kernel durations inside the full sequence can separate rank loss, preparation costs and pair interaction from noise; isolated-stage sums are not assumed exact. Keep the chosen device/event timing definition consistent, because default `run_perftest` reports device trace duration sums rather than all invocation wall time. These are targeted checks, not an additional full calibration matrix or new hard performance target.

The completed [cross experiment](../../docs/mxfp4_cross_measurement/20261010/README.md) observed different kernel-sum and trace-span winners, plus independent-versus-pipeline G2 rank differences. Preserve the main-compatible kernel-sum result definition, exclude synthetic annotation events when collecting diagnostic traces, and report a matched elapsed/span companion separately rather than calling kernel sum serving latency. A graph/event measurement is a different execution contract and must be labeled; the present experiment did not validate graph execution or AOT.

No measured total wall time exists yet. The pre-search cost record above provides an explicit range with assumptions, not a promised completion time; compare it with the observed totals. Keep targeted remeasurement for concrete doubts and do not automatically extend the matrix or add a calibration driver that stops normal shape work on a new hard deadline.

## Evidence and final checks

Each record states target and implementation commits, runtime GPU/architecture/CU, visible-device restriction, toolchain versions, activation/limit, stage2 flags, shape, input/route generation, measurement and cache conditions, numerical outcome, selected names, validation count, tuning wall time, production replay and AOT outcome.

Run focused policy/infrastructure/CSV/supervisor/parameter checks and the relevant GPU acceptance checks, plus repository formatting and compile checks. Final review checks both repository standards and this revised spec. Report unavailable evidence and pre-existing unrelated failures distinctly; do not carry forward the old branch's test counts or calibration as current acceptance.
