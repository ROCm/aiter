# 01: Make the existing MXFP4 tuning lifecycle safe

**What to build:** Existing prune/full invocations preserve valid shared output and input metadata, identify the actual runtime devices, continue supported rows after failures and return accurate final status.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Preserve input-only gate metadata through all/non-all and same/separate-file selection without reviving empty work. Validate unsupported requested rows before already-tuned deduplication.
- [x] Match summary results by stable shape keys, keep failed/unfinished rows visible and save valid later winners. Preserve main's existing zero/nonzero exit policy with correct accounting of failed/unfinished requested rows.
- [x] Preserve valid shared-CSV rows; filter tags before dropping their columns; infer missing gfx by row, retain valid identities, supply meaningful numeric defaults, and safely deduplicate/project canonical output even on empty runs.
- [x] Preserve other tuners' fields, numeric/NT values, timing/profile meanings and supported-device contracts. MXFP4 retains valid-winner-only output, us1=us/us2=0 and current profiling scope; remove its input-only metadata from final output and clarify its profile help.
- [x] Apply runtime-device preflight to MXFP4 prune/full, with the same contract reused by staged in 02. Validate every actually scheduled GPU before tuning/worker launch, reject unsupported participating devices and mark results with assigned runtime gfx/CU. Unused unsupported visible devices and differing build/global-agent identity alone do not cause rejection; other tuners do not inherit the MXFP4 gfx950 restriction.
- [x] Cover remapping, homogeneous/mixed devices, unsupported participating versus unused devices and runtime/build mismatch using simulated device properties and controlled tuning outcomes. Preserve prune/full candidate order, parallelism selection and scheduling/supervision mechanisms; these checks require no real GPU measurements.
- [x] Complete input→result→CSV/sort→summary/exit regressions for all/non-all, same/separate files, unsupported already-tuned shapes, empty selections/runs, shared/tagged/legacy/mixed-gfx/numeric data and ordinary-tuner compatibility.
- [x] Run a complete shared-infrastructure regression through at least one real non-MXFP4 tuner, such as ordinary fMoE or GEMM, using controlled measurement outcomes and its own CSV/summary entry points. Check successful writes, sorting/deduplication, failed/unfinished rows, partial-success persistence, empty runs, zero/nonzero command exits and its own schema/timing/profile semantics. Do not substitute only a toy base-class fixture or require extra GPU tuning for this coverage.
- [x] Deliver focused tests, relevant checks, behavior documentation and reviewable evidence against the [specification](../spec.md).

## Answer

Implemented safe input selection, unsupported-row reporting, stable summary, valid-winner-only persistence, tag/legacy/numeric/duplicate/multi-file normalization and runtime device preflight while retaining prune/full candidate order, parallelism and existing single-slot current-device versus multi-slot scheduling. Ordinary fMoE uses its own real lifecycle with controlled measurements and preserves schema, NT, stage timing/profile and exit meanings.

Implementation commit: `3a05e8ce086c857128c0ecc0b2f961bdb2f680fc`; implementer tip: `eb33ca87d0b7e65b1646f0ad366d194cdeeee1e8`; integration merge: `47cb9fb46d8ea532beea98d6dc7d920475f7ece6`.

Acceptance: 55 controlled lifecycle checks (38 MXFP4, 17 real ordinary fMoE) passed in 13.76 seconds; 55 existing shared infrastructure checks passed in 8.261 seconds; Black, Ruff, syntax and whitespace checks passed including merger verification. See [lifecycle evidence](../../../docs/mxfp4_staged_main_port/01/lifecycle.md), [ordinary regression](../../../docs/mxfp4_staged_main_port/01/ordinary-regression.md), and adjacent final logs. Final controlled runs hid all GPUs, used synthetic import-time discovery and blocked real GPU/process/measurement entry points. These are infrastructure results, not GPU numerical/deployment acceptance.

Execution limitation: early red fixtures accidentally spawned real GPU workers outside the prescribed coordination; they exited and were excluded from all acceptance. [Incident record](../../../docs/mxfp4_staged_main_port/01/incident.md) retains the available excerpts and explicitly states missing complete raw logs, physical mapping and UTC timestamps. Guards and offline final checks corrected the test isolation; no extra GPU tuning is required for ticket 01.
