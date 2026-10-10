# 01: Make the existing MXFP4 tuning lifecycle safe

**What to build:** Existing prune/full invocations preserve valid shared output and input metadata, identify the actual runtime devices, continue supported rows after failures and return accurate final status.

**Blocked by:** None (can start immediately).

**Status:** claimed

- [ ] Preserve input-only gate metadata through all/non-all and same/separate-file selection without reviving empty work. Validate unsupported requested rows before already-tuned deduplication.
- [ ] Match summary results by stable shape keys, keep failed/unfinished rows visible and save valid later winners. Preserve main's existing zero/nonzero exit policy with correct accounting of failed/unfinished requested rows.
- [ ] Preserve valid shared-CSV rows; filter tags before dropping their columns; infer missing gfx by row, retain valid identities, supply meaningful numeric defaults, and safely deduplicate/project canonical output even on empty runs.
- [ ] Preserve other tuners' fields, numeric/NT values, timing/profile meanings and supported-device contracts. MXFP4 retains valid-winner-only output, us1=us/us2=0 and current profiling scope; remove its input-only metadata from final output and clarify its profile help.
- [ ] Apply runtime-device preflight to MXFP4 prune/full, with the same contract reused by staged in 02. Validate every actually scheduled GPU before tuning/worker launch, reject unsupported participating devices and mark results with assigned runtime gfx/CU. Unused unsupported visible devices and differing build/global-agent identity alone do not cause rejection; other tuners do not inherit the MXFP4 gfx950 restriction.
- [ ] Cover remapping, homogeneous/mixed devices, unsupported participating versus unused devices and runtime/build mismatch using simulated device properties and controlled tuning outcomes. Preserve prune/full candidate order, parallelism selection and scheduling/supervision mechanisms; these checks require no real GPU measurements.
- [ ] Complete input→result→CSV/sort→summary/exit regressions for all/non-all, same/separate files, unsupported already-tuned shapes, empty selections/runs, shared/tagged/legacy/mixed-gfx/numeric data and ordinary-tuner compatibility.
- [ ] Run a complete shared-infrastructure regression through at least one real non-MXFP4 tuner, such as ordinary fMoE or GEMM, using controlled measurement outcomes and its own CSV/summary entry points. Check successful writes, sorting/deduplication, failed/unfinished rows, partial-success persistence, empty runs, zero/nonzero command exits and its own schema/timing/profile semantics. Do not substitute only a toy base-class fixture or require extra GPU tuning for this coverage.
- [ ] Deliver focused tests, relevant checks, behavior documentation and reviewable evidence against the [specification](../spec.md).
