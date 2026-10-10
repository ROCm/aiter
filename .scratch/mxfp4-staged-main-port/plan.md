# MXFP4 staged main-port implementation plan

Status: resolved

Reviewer decisions are incorporated and the user approved the four-ticket breakdown on 2026-10-10. The [published tickets](issues/README.md) are the sole current work set. The user subsequently authorized pre-implementation documentation corrections, including restoring the [specification](spec.md) to `ready-for-agent`; implementation is active and its state is recorded in [map.md](map.md). The spec defines behavior and the [work map](map.md) tracks ticket state. Detailed checks and the 27-search matrix are in [acceptance.md](acceptance.md).

## Integration approach

Start an isolated branch/worktree from target `882ea79dd7cc7091f859a5bedac914523ac77bbd`. Preserve the existing source working tree and its local documents. Port the resulting staged behavior and required repairs into main's interfaces instead of replaying all 16 source commits: the source's native GEMM2 assumptions and its add/revert pruning pair do not describe the accepted target behavior.

Reuse the source's policy and supervisor code where it matches the agreed contracts. Keep main's candidate enumeration, production stage calls, activation/limit handling, throughput calculations, ordinary-tuner `nt` output, and prune/full scheduling mechanisms. Add the agreed preflight to all MXFP4 modes for actual participating runtime devices; differing build/global identity alone does not reject a supported device, and other tuners keep their support contracts. Replace the source's kernel-name gate-layout inference with explicit runtime input metadata.

## Approved work sequence

| Published ticket | Blocked by | Complete delivery and verification |
| --- | --- | --- |
| 01: Safe existing tuning lifecycle | None | Input/CSV/summary/strict exit and scheduled runtime-device preflight through existing prune/full behavior, with complete lifecycle regressions. |
| 02: Complete default staged search | 01 | CLI, admitted candidate plan, references/screening, shortlist/direct exhaustive fallback/cache and fresh Pipe supervision through winner CSV and exact default production/AOT replay. |
| 03: Configured execution and AOT | 02 | Existing tuner limit environment, FP8 reduce, FP4 scatter and combinations agree through measurement/reference/validation and production/precompiled execution. The caller owns winner-file/settings consistency; no sidecar or automatic binding enforcement is added. |
| 04: Prescribed acceptance and calibration | 03 | Remaining BM/epilog/configuration evidence, 27-search outcome reporting including missing winners, elapsed diagnostics, docs and final two-axis review. |

Retain the approved ticket sequence. Parts of 03, such as prune/full parameter forwarding and AOT job generation, do not technically require staged search, but their shared edits and complete configured-stage verification stay under 03 after 02. Do not create partial-ready tickets or change the blocking graph through internal implementation steps.

Tests belong with each behavior slice. Run focused checks as changes land; run the combined relevant suite after integration. Broader GPU suites are appropriate only if their extra coverage resolves a concrete remaining regression concern.

In 01, exercise shared CSV/write/sort/summary/exit behavior through at least one real non-MXFP4 tuner with controlled measurement outcomes, alongside simulated runtime-device checks for MXFP4 prune/full. Preserve main's existing failure exit policy and each other tuner's schema/timing/profile meanings; these regressions need no extra GPU tuning.

Within 02, deliver separately reviewable/committable steps: CLI/candidate planning; screening/reference/fallback/cache with fake outcomes; process supervision with fake shape functions and coordinated events; then real GPU integration and winner/production/default AOT evidence. Apply idle, physical-identity and independent-process discipline at this first real GPU step. All steps remain one complete ticket.

Within 03, configure the environment before invocation and keep the parent/worker limit and stage2 variables unchanged throughout the tuning run. Check the inherited effective environment matches the snapshot and actual production path, because production still reads output flags per call. The caller owns CSV/settings association: a limit 5 run can skip a same-key default result without `--all`, while a valid retuned result replaces it with `--all`.

Before 04's calibration, enumerate stage/pair counts without GPU launches and plan the three full anchors' wall time using per-pair attempt overhead observed during prescribed focused acceptance. Record compilation assumptions and uncertainty before each full anchor. Keep the 27 searches, and ensure mode/configuration comparisons actually run rather than deduplicate against earlier results; cost planning adds no hard deadline or automatic scope reduction.

## Completed design experiment

The [GPU7 crossed-pair experiment](../../docs/mxfp4_cross_measurement/20261010/README.md) completed 81 separate-process measurements and a focused same-process companion, with fixed default A4W4 shape/routes. Its results support retaining the present BM/accumulation partition and end-to-end winner selection; no automatic merge of atomic/non-atomic or additive performance assumption is introduced. The merged shortlist retained the sample kernel-sum best, but could discard the trace-span best. The smaller companion exposed execution-condition sensitivity, not an isolated cache cause.

Tickets 02–04 must document the kernel-sum metric, check matched production execution and report elapsed companions when interpreting latency. The 27-search matrix, supported non-default environments, numerical gate and absence of hard performance goals remain unchanged. This experiment is prior design evidence, not replacement acceptance for the integrated port or AOT.

## Implementation tickets

See [map.md](map.md) for the approved four-ticket set, statuses and blocking edges. Prior seven/ten drafts and the approved four-ticket snapshot are archived and cannot be claimed. Each published ticket includes its own behavior checks and focused production evidence; the completed design experiment does not replace integrated-port acceptance.

For every real GPU acceptance/calibration task, use acceptance's physical-card identity and visibility procedure, then launch a fresh process only after idle checks. The current target maps by UUID and verified PCI to logical0. Compared tasks stay on the same physical card and execute sequentially.

Implementation and required current acceptance are complete on `integration/mxfp4-staged-main-port`; all four tickets are resolved. See the ticket Answers, map and [final delivery](../../docs/mxfp4_staged_main_port/DELIVERY.md) for commits, evidence and remaining operational limits.
