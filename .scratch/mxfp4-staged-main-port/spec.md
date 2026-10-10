# MXFP4 staged search port to main

Status: claimed

Published: 2026-10-10 through the user's explicit `to-spec` request. This specification consolidates the confirmed design, pipeline audit and single-GPU experiment. It defines the port to main `882ea79dd7cc7091f859a5bedac914523ac77bbd`; implementation state is maintained in [map.md](map.md).

Revision state: reviewer decisions are incorporated. Reference/all-stage-screening failure now uses group-local exhaustive validation. The user assigns CSV/configuration consistency to the caller; no sidecar or automatic configuration binding is required. The user approved and published four implementation tickets on 2026-10-10. Implementation is authorized and active; the [ticket map](map.md) records completed acceptance and the current frontier. See [reviewer-comments.md](reviewer-comments.md) and the [ticket map](map.md).

## Problem Statement

Main's MXFP4 FlyDSL tuner validates every compatible GEMM1/GEMM2 pipeline pair in its selected candidate universe. Its default M_est pruning reduces GEMM1 configurations but can still leave hundreds of end-to-end validations per shape. Users need an additional measured-screening strategy that reduces expensive pair validation while preserving main's default commands and deployable tuned results.

The earlier staged implementation cannot be replayed unchanged onto main. Main now uses layout-v2-only GEMM2 candidates, runtime gate metadata and evolved execution interfaces. The port must retain its search and recovery behavior, repair known CSV/device/metadata problems, and align screening, validation, production and AOT under the same effective configuration.

The candidate graph supports Cartesian compatibility groups, but independent stage timings do not exactly determine fMoE latency or pair correctness. The crossed-pair experiment supports proceeding with an opt-in shortlist strategy and retaining normal end-to-end selection; it does not establish a universal optimum or guaranteed speedup.

## Solution

Extend the existing MXFP4 search selector with `staged`, keeping `prune` as the default. Staged first uses main's M_est-pruned candidate universe, independently screens stage candidates within compatible contexts, then validates shortlisted pipeline pairs end to end. If reference bootstrap or stage screening cannot produce a shortlist, or its shortlist has no valid finite result, exhaust the admitted pairs of that group through normal end-to-end validation, reusing all returned outcomes.

Keep one valid winner per shape in the existing production CSV. Preserve strict unsupported-row reporting and batch continuation, staged-only process isolation, operation-level soft timeouts and accurate runtime-device identity. Support default execution, FP8 reduce, FP4 scatter and configured swiglu_limit, including matching precompiled jobs and explicit production replay.

Validate the completed port through policy/lifecycle tests, single-card numerical and deployment checks, and the agreed 27-search calibration matrix. Report performance rather than imposing a hard regret or speedup target.

## User Stories

1. As an existing tuner user, I want omitted search options to retain main's prune behavior, so that existing commands continue to work.
2. As a calibration user, I want explicit full search to exhaust main's unpruned legal candidate universe, so that I can assess M_est pruning separately.
3. As a tuner user, I want to select staged using the existing search parameter, so that I have one clear search interface.
4. As a tuner user, I want staged to screen M_est-pruned candidates, so that heuristic and measured reductions combine.
5. As a maintainer, I want all modes to share the pipeline-pair enumeration, so that their candidate universes cannot drift through independent registries.
6. As a maintainer, I want staged planning to perform no GPU pair execution, so that enumerating records does not consume validation work.
7. As a tuner user, I want any positive screening top-K, so that I can choose shortlist breadth explicitly.
8. As a tuner user, I want screening top-K distinguished from routing top-k, so that CLI and measurements are unambiguous.
9. As a tuner user, I want invalid CLI combinations rejected before tuning, so that a run cannot be mislabeled.
10. As a maintainer, I want native GEMM2 and re-tile candidates to remain excluded, so that the port follows main's supported candidate model.
11. As a tuner user, I want valid layout-v2 shapes with non-256 inter dimensions to remain eligible, so that obsolete native alignment rules do not discard them.
12. As a correctness engineer, I want fixed and derived pipeline contracts checked before grouping, so that incompatible intermediates never enter a Cartesian shortlist.
13. As a performance engineer, I want GEMM1 screening reused for compatible output accumulation modes, so that the same stage work is not repeated unnecessarily.
14. As a tuner user, I want atomic and non-atomic GEMM2 shortlists retained separately, so that one output mode cannot eliminate all configurations of the other during screening.
15. As a tuner user, I want reduce and scatter compared through complete GEMM2 output handling, so that their different final operations are fairly included.
16. As a performance engineer, I want GEMM1 screening to time its actual kernel, so that setup work does not distort stage rankings.
17. As a correctness engineer, I want deterministic, checked reference producers and consumers, so that independent GEMM2 screening receives a layout-faithful intermediate.
18. As a maintainer, I want reference bootstrap kept separate from winner validation, so that untimed fixtures cannot become tuned results.
19. As a tuner user, I want successful stage candidates ranked stably, so that equal screening scores produce deterministic shortlists.
20. As a tuner user, I want a group with no usable reference or shortlist, or an invalid shortlist, to exhaust its admitted pairs, so that independent fixture failures cannot prevent normal pair validation from finding a valid winner.
21. As a tuner user, I want returned screening failures to remain reachable during pair fallback, so that stage screening is not a permanent blacklist.
22. As a maintainer, I want all returned pair outcomes cached per shape, so that fallback never reruns the same pipeline pair.
23. As a tuner user, I want every group with admitted pairs to remain eligible for end-to-end validation and winner selection, so that reference or screening failure does not silently discard a pipeline mode.
24. As a correctness engineer, I want the actual selected G1/G2 pair checked numerically, so that reference correctness is not mistaken for all-pair correctness.
25. As an operator, I want unsupported or empty shapes reported and later supported shapes processed, so that one failed row does not discard the batch.
26. As an operator, I want failed or unfinished requested rows to produce a nonzero final command status, so that automation can detect incomplete tuning.
27. As a retuning user, I want input-only gate metadata preserved through selection and deduplication, so that unsupported layouts cannot be silently treated as already tuned.
28. As a configuration consumer, I want valid historical rows preserved in shared CSVs, so that MXFP4 tuning does not corrupt unrelated configurations.
29. As a configuration consumer, I want tag filtering, legacy device identities and numeric defaults handled correctly, so that production can still read merged results.
30. As a configuration consumer, I want the existing CSV schema and timing meanings preserved, so that deployment readers require no migration.
31. As an operator, I want each staged shape isolated in a fresh worker even on one GPU, so that fatal worker death affects the current shape.
32. As a performance engineer, I want per-GPU measurement exclusivity within one tuning run, so that independent measurements do not overlap on the same logical GPU.
33. As an operator, I want results retained until worker exit and resources closed promptly, so that GPU reassignment and large batches remain reliable.
34. As an operator, I want cancellation and unexpected IPC failures cleaned up and propagated, so that partial failure cannot leave owned workers running unnoticed.
35. As an operator, I want timeout help to describe actual soft-timeout behavior, so that I understand that a still-live permanent GPU hang is not automatically recovered.
36. As an MXFP4 tuner user in prune, full or staged mode, I want every scheduled runtime GPU validated and results marked with its identity, so that unsupported participating devices are rejected and visible-device remapping cannot mislabel configurations.
37. As a deployment engineer, I want FP8 reduction to use production-equivalent routing weights, so that tuning measures the path inference will execute.
38. As a deployment engineer, I want FP4 scatter payload/scales and final reduction included in tuning and AOT, so that its lossy execution path is correctly validated and precompiled.
39. As a model user, I want configured activation limits used throughout screening and reference checks, so that the tuner optimizes my actual activation contract.
40. As a deployment engineer, I want AOT and production to receive the same effective limit and output settings, so that the tuned names describe the compiled execution.
41. As a correctness engineer, I want all output modes to use the configured numerical gate and reject non-finite errors, so that a fast invalid result cannot win.
42. As a deployment engineer, I want production replay to verify exact names and the actual input/quantization path, so that heuristic fallback cannot masquerade as successful deployment.
43. As a calibration user, I want all comparisons on one idle-checked physical card in sequential independent test processes, so that card variation and overlapping tasks do not confound results.
44. As a calibration user, I want pruning and screening losses reported separately, so that existing M_est pruning is not attributed to staged search.
45. As a calibration user, I want kernel-duration and elapsed measurements labeled separately, so that GPU work savings are not misrepresented as serving latency.
46. As a calibration user, I want reproducible calibration and measured ranking discrepancies retained, so that future strategy changes are based on evidence.

## Implementation Decisions

### Base, scope and interface

- Port the staged behavior and necessary repairs onto the fixed main base in an isolated integration branch. Preserve the source working tree and local material. Adapt to main's interfaces rather than replaying all source commits, including its canceled pruning change.
- The supported tuner contract is gfx950, BF16 caller input/output, A4W4 with per-1x32 quantization, G1U1, separated gate layout and no stage1 routing weight. Support Silu, Swiglu and Situv2 with existing default Situv2 parameters and production-valid limit behavior. Other input contracts are unsupported rows, not implicit alternative-tuner requests.
- Keep `--mxfp4-flydsl` as tuner selection. Extend `--mxfp4-search-mode` to `prune|full|staged` with default `prune`.

| Mode | GEMM1 candidate universe | Pipeline-pair evaluation |
| --- | --- | --- |
| prune | Main's M_est-pruned configurations | Exhaustive within that universe |
| full | All statically legal main configurations | Exhaustive within that universe |
| staged | Main's M_est-pruned configurations | Stage screening, shortlist validation and group fallback |

- Preserve prune/full candidate order, parallelism selection and scheduling/supervision mechanisms. Runtime-device preflight applies to MXFP4 prune and full as well as staged, as an explicitly authorized device repair; unsupported participating devices are rejected before work. Shared correctness, metadata, CSV and effective-parameter repairs also apply where needed; unrelated full-search supervision remains outside the port.
- Add `--mxfp4-screen-topk`, default 2, accepting any positive integer. An explicitly supplied value is valid only with explicit staged mode. Retain main's CLI checks for tuner/grouped combinations and fail before shape work for invalid invocation settings.

### Candidate universe, identities and groups

- Derive staged candidates solely from the same main pair enumerator used by prune. Constructing candidate records performs no GPU pair execution. Fallback remains within this pruned universe and cannot recover M_est-excluded configurations.
- Keep main's layout-v2-only GEMM2 enumeration and equal sorting/G1/G2 BM constraint. Do not restore native GEMM2, native alias filtering, or re-tile configurations. Admit non-256 inter dimensions when statically supported layout-v2 pairs remain.
- Validate actual gate/activation/dtype/quantization contracts, G1 intermediate payload/scales and sorting BM before grouping. G1 uses inline input quantization at BM16 and fused-prequantized input at BM32/64/128; BM16 native scale layout is derived, not a free tuning parameter.
- GEMM1 screening context is BM. GEMM2 screening context and pipeline compatibility group are BM plus accumulation mode: atomic versus non-atomic. Reduce and scatter share the non-atomic group because complete G2 execution consumes the same admitted intermediate and overwrites final BF16 output.
- Preserve all effective stage launch distinctions as identities, including epilog, tiles, loads, persistence, waves, k_wave, XCD and prefetch. The pipeline-pair identity contains both complete stage identities, not a group key. Use the domain terms PipelineCompatibilityKey, PipelinePairKey and StagedSearchPlan where named types are needed.
- Shape, runtime device, routing data, activation/limit and effective output environment are fixed for one screening/validation context. Never reuse scores or pair outcomes across those contexts.
- These are compatibility groups, not a guarantee of exact performance separability. Independent stage scores rank a shortlist; final correctness and latency belong to normal end-to-end pair validation.

### Screening and reference bootstrap

- Screen G1 once per BM and reuse its ranking across accumulation modes. Time only the actual G1 kernel. BM16 inline quantization is included; sorting, external input quantization, tensor/buffer preparation and reference checks remain untimed setup.
- Each G2 screening attempt creates its own intermediate during untimed setup using a validated reference producer. Keep the intermediate candidate-local; release it before another stage candidate or BM setup.
- Time complete G2 execution through final BF16 output. Include atomic output initialization or non-atomic reduction/scatter, effective payload/scales and required route weighting. Preserve production-equivalent parameters and identify differences between isolated and in-pipeline execution honestly.
- Use existing warmup/iteration settings for screening; no reduced screening schedule is introduced.
- For each BM, select reference producers only from that shape's admitted M_est-pruned G1 stage candidates for the same BM. Prefer BN128, BK256, four waves, k_wave1, no hidden prefetch/XCD and cached loads; BM16 uses its required non-temporal variant. If the preferred configuration is absent, use the admitted candidates in original order; after a returned failure, try all remaining untested candidates in that same admitted set. Bootstrap does not enumerate full/unpruned or cross-BM configurations.
- Select reference consumers only from admitted atomic layout-v2 G2 candidates for the same shape/BM. Prefer N128/K128, non-persistent execution and cached loads when admitted. Self-test on a Torch-produced layout-faithful intermediate, then try remaining admitted atomic consumers in original order.
- A chosen producer/consumer pair must pass one untimed end-to-end correctness check. Bootstrap is a correctness fixture, not a timed validation run, cache entry or automatic winner. If no reference validates, skip the dependent independent screening and send each affected group directly to exhaustive validation of its admitted pruned pairs; normal pair execution does not depend on the failed screening fixture.
- Successful screening requires a returned positive finite timing without catchable exception or soft timeout. Sort successful stage candidates by score then original candidate order, retaining at most K; exact ties do not increase K. If either stage has no successful screening candidate for a group, send that group directly to admitted-pair exhaustive validation without rebuilding a second candidate source.

### Pair validation, fallback and failure outcomes

- Validate the Cartesian product of retained G1/G2 stage candidates in every usable group through main's normal end-to-end execution and numerical check. Sorting, input preparation, G1, G2 and final output handling are part of that pipeline.
- Exhaustive group fallback applies when reference bootstrap failed, either stage has no successful screening, or the constructed shortlist has no valid finite pair. Evaluate every admitted pair not already present in the pair cache, including screening-returned failures. Continue after the first valid fallback to select the fastest valid group result.
- Cache every pair-validation outcome that returns control per shape, including success, numerical failure, catchable exception, soft timeout and non-finite timing. Reuse it during expansion and never rerun that identity. Store scalar outcome/timings/failure text only.
- Process every group containing admitted pairs, through shortlist validation or exhaustive fallback, and select the fastest valid end-to-end result across group winners. Reference correctness does not certify other G1/G2 pairs; check their actual produced intermediate and final result.
- The numerical gate remains main's configurable `--errRatio`, default cosine_diff ≤0.1, requiring finite error. FP4 scatter uses the same Torch reference, without automatic tolerance widening or a quantized reference that conceals additional error. Final winner timing must be positive and finite.
- Unsupported/empty shapes and shapes with no valid group winner produce a concise invalid result. Keep later supported rows progressing. A fatal worker death fails the whole shape and does not create a cacheable pair outcome.

### Execution settings and AOT

- Support default output handling, `AITER_FLYDSL_STAGE2_FP8=1`, `AITER_MXFP4_INTERMEDIATE=1`, non-default swiglu_limit and supported combinations. Final caller-visible output remains BF16; effective internal output format and weight placement follow production.
- Configure the tuner limit through the existing `AITER_MXFP4_TUNE_SWIGLU_LIMIT`: unset or empty selects activation defaults; an explicit value is parsed and checked against production-valid activation/limit combinations. Read and normalize the invocation's effective configuration once and carry that snapshot to workers, screening, bootstrap and validation. AOT receives the corresponding `--swiglu-limit`, and production receives its explicit API value; default normalization must agree at every entry point.
- Production wrappers continue to read `AITER_FLYDSL_STAGE2_FP8` and `AITER_MXFP4_INTERMEDIATE` on each call. The parent and workers must keep these variables and `AITER_MXFP4_TUNE_SWIGLU_LIMIT` unchanged from snapshot capture until the tuning run ends. Each worker's inherited effective settings must match the snapshot, including unset/empty defaults; tests verify that the actual production output path agrees. Configure separate tuning, AOT and replay invocations before starting them. This stable-environment contract adds no production configuration-binding or mutation-detection feature.
- Propagate topk_weights into FP8 reduce screening and normal pair validation so routing weights are applied exactly once in the production-equivalent final reduction. Preserve default BF16 operation when disabled.
- Apply production-equivalent limit normalization in direct G1 launches, wrappers, bootstrap and Torch references. Retain supported activation/limit combinations; configured limits do not introduce custom Situv2 beta support.
- Generate matching FP4 scatter AOT jobs with packed target/scales, dtype, persistence and eligibility. Preserve existing BF16/FP8 paths and the shared AOT code's supported bias behavior.
- Add MXFP4 AOT `--swiglu-limit`, retaining existing omitted-argument callers, and document how standalone/batch AOT receives the same limit used during tuning. Production continues to receive its explicit activation API parameters. Calibration and validation artifacts record their effective configuration outside the winner CSV.
- The caller that selects tuning output and deployment files is responsible for associating the correct winner CSV with its actual limit and stage2 output switches, avoiding reuse/overwrite of another configuration's results, and supplying matching settings to tuning, AOT and production. Existing names and CSV shape keys do not encode those values. Per the user's decision, this port adds no JSON sidecar, automatic file binding, CSV-key expansion or configuration-mismatch rejection; it documents the caller obligation. Calibration still uses the separately recorded configuration outputs specified below.
- Document the concrete reuse hazard: if a CSV already contains a same-key shape tuned at default limit, a limit 5 invocation without `--all` can silently skip it as already tuned; with `--all`, a valid new winner replaces that same-key result. Output-switch changes have the same issue. Callers must choose the intended configuration's file and avoid reusing incompatible rows; `--all` is retuning, not configuration isolation.
- Exact names alone are insufficient deployment evidence. Match actual input quant dtype, metadata, sorting/quantization path, intermediate contract and full final-output handling. Explicit A4W4 replay must not silently select default A16W4 for small Swiglu or Situv2.
- Validate matching precompiled entries and exact selected production dispatch. Success using heuristic/fallback kernels or a different compiled key does not satisfy deployment acceptance.

### Runtime devices for all MXFP4 search modes

- Before MXFP4 prune, full or staged tuning starts, validate every visible runtime GPU that its existing scheduling path will actually use. Reject the run before measurements or worker launch if any participating device does not support the gfx950 MXFP4 tuner contract. Visible devices outside the participating set do not cause rejection; this check does not extend the MXFP4 architecture restriction to other tuners.
- Determine support and each result's gfx/CU from the assigned runtime device. A differing build target or global rocminfo result alone is not grounds for rejection and must not relabel a supported device. Preflight adds a support check without replacing prune/full parallelism selection or supervision.

### Staged process supervision

- Every staged shape runs in a fresh process, including on one GPU. Within a tuning run, assign at most one active shape worker to a logical GPU; work on different supported GPUs may overlap. Cross-run exclusivity remains the caller's responsibility.
- Staged retains main's `--mp` selection: for a nonempty current batch, normalize the requested parallelism as main does, cap it by visible GPU count and batch shape count, and use at least one slot. Schedule on the first selected visible logical GPUs and preflight that participating set before launching work. With one slot, staged still uses a fresh process per shape. Empty batches launch no worker. If no visible device is available, fail preflight; unsupported participating devices are rejected under the all-mode device contract above.
- Give each worker a private one-way result Pipe and at most one final shape result. Treat result arrival and process exit separately: retain the result, confirm exit, join, inspect exit status, close resources, then release the GPU. Bound outstanding Process/Pipe resources by active workers.
- Returned Python shape exceptions and worker exits without results become shape failures; later work continues after the old worker exits. In IPC handling, clean EOF may mean no result; unexpected receive/deserialization errors fail the supervisor.
- Parent cancellation or supervisor failure stops new launches, cleans owned workers/endpoints/handles and propagates the original exception. The shared batch layer may record the exception and still run its final summary.
- Preserve operation-level best-effort SIGALRM timeout, default 1800, with owned handler/timer restoration. Add no startup, shape-wide or parent hard deadline; a still-live worker that cannot return control may wait indefinitely. Keep prune/full supervision separate from this staged supervisor.

### Input, CSV, summary and timing semantics

- Preserve input-only gate metadata through all/non-all and same/separate-file retune selection without reviving empty selections. Validate unsupported rows before already-tuned deduplication; matching shape keys cannot suppress an unsupported requested layout.
- Shared summary matches stable shape keys and does not require input-only metadata in tuned results. Preserve main's existing nonzero exit on failed or unfinished rows; repair their accounting rather than introduce a new exit policy. Supported later winners persist, but any failed or unfinished requested row produces a nonzero final command status, including mixed input CSVs.
- Preserve valid old rows when merging/sorting shared output. Apply tag filtering before removing tag columns, infer missing legacy gfx by row while retaining valid gfx, supply meaningful numeric defaults, then deduplicate and project at the safe canonical boundary. Empty runs must not corrupt existing configuration files.
- Preserve other tuners' own schemas, numeric/NT fields, timing meanings, profiles and supported-device contracts when repairing shared infrastructure. For the MXFP4 tuner, final CSV only updates valid winners; failed shape details remain in logs/summary, end-to-end latency is written to us1 and us, and us2 remains zero. Screening times never become final stage timings.
- Keep current profiling scope: no stage-ranking or full-pair profile artifacts, no new final CSV columns. Clarify the dedicated MXFP4 profile-file limitation while preserving other tuners' behavior.
- Preserve the main-compatible kernel-duration-sum definition for tuned latency and label it explicitly. Diagnostic traces count real kernel events, excluding synthetic GPU annotations. Record matched elapsed/span companions separately when interpreting deployment latency; they are distinct measurements, not interchangeable screening scores.

## Testing Decisions

### Behavioral seams and prior art

- Use the existing CLI and shape-level tuner behavior as the primary search seam. Inject fake enumeration/screening/validation outcomes through existing internal call points; assert externally observable pair selection, attempt order/count, returned outcome and later-shape continuation rather than private dataclass representation.
- Exercise input selection through CSV write/sort/summary and final exit status, not only post-processing. Include at least one real non-MXFP4 tuner, such as ordinary fMoE or GEMM, through its own entry points with controlled measurement results. Cover successful writes, ordering/deduplication, failed and unfinished rows, partial-success persistence, empty runs and zero/nonzero command exits; verify its own schema and timing/profile semantics. This infrastructure regression need not launch GPU kernels. Reuse tuner-infrastructure, retuning and comparison test conventions.
- Test staged process ownership with coordinated events/fake contexts and the dedicated supervisor boundary. Reuse multiprocessing test conventions, while covering the dedicated Pipe lifecycle rather than assuming generic worker tests cover it.
- Use the established production metadata/configuration lookup and subprocess replay seams for exact names, actual dtype/path and numerical output. Reuse current named-dispatch fixtures, tuning pipeline subprocess patterns and AOT discovery/compile entry points.
- Prefer these existing high-level seams; no new public module/interface is required. Pure policy checks execute without GPU kernels; full-module imports may still require ROCm/FlyDSL, and that requirement must be documented accurately.

### Required functional coverage

- CLI/default preservation; staged and prune share the same admitted pair source/order; static contract validation; native/re-tile exclusion; supported I384/K128; M_est boundaries 3/4, 15/16, 32/33, 63/64/65 and 128/129.
- K values 1, 2, greater than available, zero and negative; ties; G1 ranking reuse; atomic/non-atomic separation; reduce/scatter complete-output competition.
- Reference retries, bootstrap isolation, all-reference/all-stage-screening failure, invalid/non-finite shortlists, exhaustive fallback and its fastest valid result, pair-cache reuse, empty/no-winner outcomes and fatal shape failure. With deterministic returned validation outcomes, prove a valid prune pair is found by staged even when independent fixtures fail, including all atomic reference consumers failing while a non-atomic pair remains valid. Fatal worker death or a live permanent hang is not a returned fixture failure.
- Result-before-exit, exit-before-read, EOF, delayed/fatal exit, bounded handles, same-GPU ownership, multi-device simulated parallelism, cancellation/IPC cleanup, timeout restoration and prune/full supervisor isolation.
- MXFP4 prune/full/staged runtime architecture/CU and physical-to-logical mapping fixtures; reject unsupported participating devices before work, tolerate unused unsupported visible devices, and derive correct result identity even when runtime/build information differs. Other tuners retain their own support contracts.
- All/non-all, same/separate-file, supported/unsupported gate input, already-tuned unsupported rows, empty selections/runs, shared/tagged/legacy/mixed-gfx CSVs, numeric defaults, ordinary-tuner schema, partial success persistence and strict final exit status.
- Default/custom limits, worker inherited-environment/snapshot parity, actual production output-path agreement, clamp behavior, exactly-once route weighting, FP8 reduce/FP4 scatter and combined flags, packed output/scales, unchanged atomic behavior and matching AOT keys/entries. Tests use a stable environment throughout each invocation.

### Single-card numerical and deployment checks

- Before every GPU test task, check the target card is idle. Each comparison task runs in a fresh independent process, GPU tasks execute sequentially, and all compared measurements use the same physical card. Test-run scheduling must not overlap another experiment task on that card.
- Select one physical GPU for each comparison and verify its PCI/UUID after visibility mapping; monitor and HIP ordinals must not be assumed equal. Record idle samples, task process start/exit and identity, and run all compared tasks on that same verified card. The current host selection and commands are documented in acceptance; multi-device tuner logic uses simulated fixtures during single-card checks.
- Use default MiniMax pairs at token128 and token2048 with H3072/I384/E256/routing top-k8 to cover admitted BM16/32 and BM64/128 respectively, atomic/reduce, and BM128 scatter explicitly.
- Use GPT-OSS Swiglu pairs at token128 and token2048 with H3072/I512/E128/routing top-k4 to cover K128/K256, FP8 reduce, FP4 scatter N128/N256, limit5 and both stage2 flags with default/custom limit. Pair selection is a test fixture, not an alternative candidate registry.
- Retain the small token4/H3072/I512/E128/top-k4 staged winner smoke and a default-parameter Situv2 replay using a supported tuple and explicit A4W4 contract. The small smoke cannot establish BM128 scatter coverage.
- Require finite passing numerical results, correct timing/schema, exact selected production names/path and matching precompiled entries. Record toolchain versions; unavailable compile/GPU evidence remains incomplete.

### Calibration and performance reporting

Run 27 initial shape-mode searches with default screening top-K 2, consistent warmup/iterations, matching input/routes, controlled compilation/cache conditions and separate outputs for each effective configuration. Ensure each prescribed search actually runs instead of deduplicating against another mode's already-tuned rows.

Before calibration, enumerate admitted stage and pair counts without launching GPU kernels, especially for the three full anchors below. Record per-BM/accumulation counts and a wall-time range for each full anchor before starting it, using representative per-pair attempt overhead from the prescribed focused checks and explicit cold/warm compilation assumptions. Include data/reference preparation, warmup, profiling and failure costs; winner kernel latency times pair count is not a tuning-time estimate. Record uncertainty and update estimates with observed progress; static pair counts do not establish runtime validity. This cost planning does not add searches, hard deadlines or automatic matrix reduction.

Default MiniMax Silu uses fixed H3072/I384/E256/routing top-k8:

| Tokens | M_est | Searches |
| --- | --- | --- |
| 64 | 2 | full, prune, staged |
| 128 | 4 | prune, staged |
| 512 | 16 | prune, staged |
| 1024 | 32 | prune, staged |
| 2048 | 64 | full, prune, staged |
| 4096 | 128 | prune, staged |
| 8192 | 256 | full, prune, staged |

These are 17 searches. Extended GPT-OSS Swiglu uses fixed token2048/H3072/I512/E128/routing top-k4 and runs prune/staged for each configuration:

| Configuration | FP8 reduce | FP4 scatter | Limit |
| --- | --- | --- | --- |
| Default | Off | Off | Default 7 |
| FP8 | On | Off | Default 7 |
| FP4 | Off | On | Default 7 |
| Custom limit | Off | Off | 5 |
| Combined | On | On | 5 |

These add 10 searches. Cover both-flags/default-limit with specified pairs only; that combination does not add another tuning invocation beyond the 27-search matrix.

- Report per-shape selected names, numerical error, validation count, kernel-sum latency, matched elapsed diagnostics and tuning wall time. Disclose full-to-prune, prune-to-staged and full-to-staged effects only where the corresponding results exist; missing/invalid references yield unavailable comparisons.
- Report existing M_est pruning and new stage-screening effects separately. Keep summed default-matrix tuning time and extended-environment results distinct. There are no hard latency-regret or speedup thresholds.
- For each paired prune/staged calibration point, record whether each mode produced a valid winner and count prune-valid/staged-invalid outcomes separately from latency regret. Classify missing-reference, screening, normal pair, worker-death and incomplete-run outcomes; do not hide invalid staged results by omitting the comparison row.
- Investigate concrete noisy/negative/anomalous outcomes through alternating replay of selected pairs on fixed data and controlled measurements. Record median/dispersion, stage ranks and shortlist retention; use small crossed/conditioned measurements where they resolve a specific discrepancy. Re-search only when ranking evidence justifies it. Do not automatically expand the full matrix or change K/fallback semantics to improve numbers.
- Keep reproducible independent validation artifacts: commits, runtime device/CU, visible mapping, effective flags/limit, data/routes, toolchain, measurement/cache conditions, counts/times, numerical and dispatch/AOT outcomes.
- Complete relevant policy/infrastructure/CSV/supervisor/parameter checks, GPU acceptance and repository formatting/syntax/diff checks. Review the final diff on Standards and Spec axes and resolve actionable in-scope findings. Historical experiment counts and known unrelated baseline failures are not substitutes for current acceptance.
- Tests exercise existing high-level tuning and deployment seams, so internal representation changes do not invalidate behavioral coverage. Keep current-port acceptance separate from historical experiments; old results establish neither completed implementation nor current production/AOT acceptance.

## Out of Scope

- Changing staged to the default, replacing main's M_est pruning, or adding a second independent search selector.
- Native GEMM2 restoration, native runtime alias canonicalization, re-tile expansion or candidates from other fMoE execution families.
- Additional architectures, A16W4/A8W4 contracts, interleaved tuning, bias/EP/padding extensions, custom Situv2 betas, or alternate input-quantization policies.
- Automatic merging of atomic/non-atomic groups, per-epilog shortlist quotas, adaptive K, tie/gap-driven expansion or a claim of exact performance additivity/global optimality.
- Cross-candidate/intermediate tensor caching, reduced screening iterations, per-pair process isolation or recovery after fatal GPU failure.
- Startup/shape-wide/parent hard deadlines, automatic termination of a still-live permanent hang, unrelated prune/full Queue redesign or system-wide cross-run GPU locking.
- Persistent failed rows, new winner CSV columns, complete pair/screening profiles, or an integrated dual/multi-search command.
- Configuration sidecars, automatic output-file/environment binding or mismatch rejection, and production runtime inference of settings from winner names. The caller owns configuration/output consistency.
- Hard performance targets, unrequested broad calibration expansion or treating this specification publication as code implementation, merge or deployment.

## Further Notes

The implementation base is main `882ea79dd7cc7091f859a5bedac914523ac77bbd`. The staged source is `3f3d5f63034ef524746b2271820a9e9ac53357f4..f68e08528cb85f7f3f7211d3f8a1664660bb7f50`; it is parallel history, not an ancestor of main. Adapt its final behavior and repairs to the selected base.

The single-card experiment is retained on `experiments/mxfp4-cross-measurement`, data commit `96e2b5545` and deterministic-analysis tip `4977e927d`. Its 81 formal results used one default-environment BM128 3×6 sample and passed consistency/numerical checks. Both separate and merged K2 retained the kernel-sum winner in three rounds, but G2/pipeline rankings differed and merging discarded the diagnostic trace-span winner. A smaller interleaved companion changed rank under changed measurement conditions. These results support an opt-in staged heuristic with normal pair validation; they establish neither safe universal merging nor exact separability. Public CSV replay covered one selected pair; graph/AOT and the integrated port were not accepted by that experiment.

Read these supporting sources when implementing their corresponding parts:

- [Design decision record](design-decisions.md): Q1–Q14 and recovered planning history. This published specification is authoritative; historical pending-confirmation text is archived.
- [Facts](facts.md) and [pipeline grouping audit](pipeline-grouping-audit.md): source/main differences, actual dispatch/quantization/output chains and measurement limitations.
- [Execution/AOT support work](environment-options.md): verified parameter/compilation gaps for FP8 reduce, FP4 scatter and configured limits.
- [Detailed acceptance](acceptance.md): fixture rationale and measurement procedures subordinate to this specification.
- [Cross-experiment review](cross-experiment-review.md) and [experiment artifacts](../../docs/mxfp4_cross_measurement/20261010/README.md): reproducible data and its bounded conclusions.
- [ADR 0001](../../docs/adr/0001-mxfp4-search-modes-on-main.md): why the single main-compatible selector stages after pruning.
- [Implementation plan](plan.md) and [ticket map](map.md): the approved four-ticket sequence and current published ticket state.
