# Kimi3 TP8 tuning handoff

Status: **unfinished; resume measurement in the new Codex environment.**
Handoff date: 2026-11-19, Asia/Shanghai.

The user asked to save all intent, changes, and requirements here before moving
Codex environments. Do not treat the preliminary selection as a finished tuning
result. No production tuning CSV has been modified by this task. The user subsequently
requested committing the existing work and this document for environment transfer.
The handoff commit includes this document and `tune_kimi_artifacts/`, preserving
all candidate CSVs, scripts, launch manifests, successful results, and failed logs.

## User intent and requirements

1. Create and work on branch `dev/kimi3_tp8`.
2. Tune all existing Kimi3 TP8 cases. The original reference URL was:
   <https://github.com/ROCm/aiter/blob/main/aiter/configs/model_configs/kimik3_gfx1250_tuned_grouped_fmoe.csv>.
3. **Do not perform a broad parameter sweep.** The user reiterated this:
   “你不需要sweep，你需要拷贝现有的case，并且简单tune tilen 256/512”.
4. Copy existing **non-EP** schedules from
   `aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv` where
   `tile_m` matches. Keep the target shape and its M tiles unchanged. Keep the
   copied schedule fixed while comparing only N=256 and N=512.
5. Benchmark against the actual current main configuration, under identical
   workload, initialization, timing, and environment settings. Confirm that
   retained changes outperform main; keep main when a candidate does not win.
6. Deliver **one full table covering GEMM1 and GEMM2 TFLOPS and bandwidth** for
   every case, with comparison against main. Prefer throughput/BW columns over
   timing columns in the final user-facing table.
7. Use the GPU-lock skill and check GPU process idleness before every launch.
   Wait while other processes exist; do not rely on GPU utilization and do not
   terminate another user's jobs.
8. Latest user hypothesis: memory-access faults **may be caused by conflict
   with other GPU processes**. This is plausible but unconfirmed. Do not label
   the copied schedules as defective merely from these interrupted runs.
9. Reply Chinese first, then English. Telegram bridge stays disabled. Do not
   spawn subagents unless newly authorized; the old `prefence_analysis` agent
   belongs to an unrelated prior task.

## Branch, baseline, and preservation

Workspace: `/app/aiter`.
Current branch: `dev/kimi3_tp8`.
HEAD/main baseline fetched at task start:
`956fd46d879582fec765f614429638b21c6200cc`.

The new branch was created from that `origin/main`, not from the previous DSV
branch. The subsequent handoff commit preserves work in progress, not a
validated production configuration change. The previous branch
`dev/dsv41_flash_ep8_ep16` remains intact, with commits:

- `3f8f10771`: inherited DSV4.1 Flash EP8/EP16 configs.
- `8a4252684`: subsequent user commit, the donor snapshot source for this task.

The working-tree submodule `3rdparty/composable_kernel` is checked out at
`96afae1d54fc79c26857de7e1683b0dde1ff3c6f`, while the new main branch records
`f93c029e09f67c983696eecac8e0e0350f3dbb54`. It appeared modified after switching
branches. Do not include that unrelated gitlink difference in the tuning change.
Several pre-existing untracked run scripts and tests remain in the workspace;
do not delete, overwrite, or commit them as part of this task.

**At handoff, the production TP8 CSV still equals main.** All candidate changes
exist only in the external artifact directory. The repository now also contains `tune_kimi_artifacts/`, a committed copy of
the external experimental files so they survive an environment transfer.

## Actual TP8 workload and scope assumption

The linked `kimik3_gfx1250_tuned_grouped_fmoe.csv` contains intermediate sizes
3072 and 768. The repository has a dedicated TP8 file:

`aiter/configs/model_configs/kimik3tp8_tuned_grouped_fmoe.csv`

The existing benchmark explicitly identifies that file as Kimi3 TP8 with:

- Hidden/model dimension: **3584**.
- Intermediate dimension per TP rank: **384 = 3072 / 8**.
- Experts: **896**, all local on the TP compute path; this is not EP.
- Topk: **16**, not the DSV4.1 Flash topk=6.
- Format: **A4W4**, BF16 output, per-1x32 MX scales.
- Activation: **SiTUv2**, beta=4, linear_beta=25; preserve this instead of
  copying the donor's SiLU activation.
- No bias, balanced routing, seed=0.
- Uniform random inputs and uniform random scales; no constant initializer.
- All 15 existing token buckets:
  `1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384`.

I sent a clarification about the linked file versus the dedicated TP8 file,
then explicitly proceeded with the dedicated file above. The user continued
steering the work but did not separately answer the shape question. This is the
working scope, supported by the existing Kimi3 TP8 test; do not silently switch
this to the intermediate=768 cases.

Preserved tile_m/tile_m2: 16 for tokens 1..512, 32 for token 1024, 64 for tokens
2048/4096/8192, and 128 for token 16384.

## Important N-width constraint

In `aiter/ops/flydsl/grouped_moe_gfx1250.py`, configured N tiles must divide N;
configured invalid widths raise rather than silently narrowing.

- GEMM1 output N = 2 * 384 = **768**. N256 is valid; **N512 is invalid**.
- GEMM2 output N = **3584**. Both N256 and N512 are valid.

Consequently both copied candidates use GEMM1 `tile_n=256`. They differ only in
GEMM2 `tile_n2=256` versus `512`. All 15 candidate pairs were checked to differ
in exactly that one field. No other scheduling dimension was swept.

## Donor mapping and candidate construction

Original artifacts root:
`/workspace/trace/kimi3_tp8_20261119/`

Committed portable snapshot: `tune_kimi_artifacts/` next to this document.
After moving environments, restore that snapshot to the original artifact path
or adjust the scripts' absolute ROOT/PYTHON paths. The scripts derive their
artifact directory from their own location, so they can also run directly from
the snapshot directory after adjusting the workspace and Python paths.
Historical launch manifests intentionally retain the exact original paths.
The venv itself is not committed; recreate FlyDSL 0.5.0 if it is unavailable.

Before switching branches, the current DSV CSV was copied to `donor.csv` there.
The donor family selected is A4W4 non-EP, expert=96, topk=6, inter_dim=2304.
For each target, choose a donor with matching `tile_m` **and** `tile_m2`, and the
closest routed rows per expert on a log scale:

- Target density: `target_tokens * 16 / 896`.
- Donor density: `donor_tokens * 6 / 96`.

This is a deterministic choice of one existing schedule, not a donor sweep.

| Target tokens | M tile | Donor tokens |
|---:|---:|---:|
| 1 | 16 | 1 |
| 2 | 16 | 1 |
| 4 | 16 | 1 |
| 8 | 16 | 2 |
| 16 | 16 | 4 |
| 32 | 16 | 8 |
| 64 | 16 | 16 |
| 128 | 16 | 32 |
| 256 | 16 | 64 |
| 512 | 16 | 128 |
| 1024 | 32 | 256 |
| 2048 | 64 | 512 |
| 4096 | 64 | 512 |
| 8192 | 64 | 512 |
| 16384 | 128 | 2048 |

Copy schedule fields from `tile_m` through `kernelName2`; preserve all workload
key fields, including SiTUv2 and target experts/topk/dimensions. Override N as
above. Reset copied `us/tflops/bw` to zero rather than carrying donor timings.
The candidates therefore change scheduling fields relative to main as a result
of copying, but the subsequent comparison tunes only N.

## Environment and reproducible benchmark

Use the existing isolated Python with **FlyDSL 0.5.0**, matching requirements:

`/workspace/trace/dsv41_flash_ep8_ep16_20261119/venv/bin/python`

The global installation is an older FlyDSL; do not use it or modify it casually.
The venv uses system packages. GPU is gfx1250, 256 CUs.

Benchmark entry point:
`op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel`.
This tests non-EP/TP through the real fused_moe dispatch and captures isolated
GEMM1/GEMM2 callables. It is not `test_moe_ep.py`, and it does not include TP
collectives or end-to-end MoE overhead.

Representative command (launch only through the lock/idle procedure):

```bash
/workspace/trace/dsv41_flash_ep8_ep16_20261119/venv/bin/python -u \
  op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel --data-format a4w4 --experts 896 \
  --tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  --topk 16 --model-dim 3584 --inter-dim 384 \
  --act situv2 --no-bias --iters 101 --no-check-aot-cache \
  --data-init uniform --scale-init uniform --balanced-routing --seed 0
```

Warmup is the script's default 5. Timing is the benchmark's existing native
`run_perftest` path (internally uses torch profiler). Do not add rocprof/ATT.
Correctness uses the existing PyTorch reference and logits_diff < 0.01 gate.

`tune.py` removes inherited AITER_/FLYDSL_/LLVM_GFX1250 tuning overrides and sets:

```text
AITER_RUNTIME_GPU_ARCH=gfx1250
AITER_USE_SYSTEM_TRITON=1
ENABLE_CK=0
LLVM_GFX1250_ANY_XNACK_CODEGEN=0
AITER_GROUPED_DEBUG=0
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
AITER_FLYDSL_TOKEN_MULTIDEST_QUANT=1
AITER_FLYDSL_MXFP4_WMMA_B2B=1
AITER_LOG_MORE=0
FLYDSL_DUMP_IR=0
AITER_MOE_EXPERT_BALANCE=true
AITER_CONFIG_GROUPED_FMOE=<artifact variant CSV>
PYTHONPATH=/app/aiter
PATH=<venv bin>:<existing PATH>
```

Per-run JSON files record exact command, environment, config, token list, and
whether the lock wrapper was installed. Logs retain native output and idle
checks. Results use the native markdown summary's greater timing precision,
not only the two-decimal stage log lines.

## GPU lock and concurrent-process uncertainty

Read the now-available skill:
`/workspace/config/agents/skills/gpu-lock/SKILL.md`.
It was found earlier at `/root/.claude/skills/gpu-lock/SKILL.md`.

The skill says:

> Use `/usr/local/bin/gpu-lock` when it is an executable file ...
> If no executable wrapper is installed, locking is not required: run the
> authorized workload directly without asking for confirmation or installing
> a replacement. Still follow gpu-idle-check immediately before each launch.

During these measurements neither an executable wrapper nor the old shared
`/var/lib/gpu-lock/amd-gpu.lock` existed in this container. The benchmark runner
therefore ran directly, as the invoked skill permits, **after** a successful
`rocm-smi --showpids` reporting `No KFD PIDs currently running` each time.

Other jobs repeatedly appeared in rocm-smi, often with host PIDs not visible in
this container's /proc. The runner waited while they existed. Without a shared
lock, the pre-launch check cannot prevent another container from starting a job
after our check. **The user now suspects this race explains the faults.**

In the new environment, check again for the wrapper and use it if installed.
Acquire the lock around the actual workload, perform the idle check inside the
locked process, hold it until the child exits, and release/retry if busy. Do
not create a private substitute lock or use utilization as an idle criterion.
`tune.py` and `idle_exec.py` already implement the installed-wrapper/fallback
policy. Missing an installed wrapper is not, by itself, a reason to ask the user
for permission or stop, per the explicit skill.

## Measurements completed and preliminary gains

`screen_main`, `screen_n256`, and `screen_n512` completed all 15 cases, with all
correctness checks passing. These compare copied schedules, not a broad sweep.

Initial best full-row selections based on GEMM1+GEMM2 summed time were:

| Tokens | Copied candidate | Preliminary combined throughput gain vs main |
|---:|---|---:|
| 1024 | G1 N256 / G2 N256 | +5.58% |
| 2048 | G1 N256 / G2 N512 | +3.10% |
| 8192 | G1 N256 / G2 N512 | +6.24% |

All other rows provisionally retained main: copied candidates were slower or
the difference was below 1%. Complete screening data for every token and stage
are in `screen_*_results.json`, with TFLOPS, bandwidth GB/s, and native us.
Do not present the screening gains as confirmed final results.

The first verification included all three changed rows. After faults at 8192,
I conservatively reverted that row **in the external selected.csv only**, so
current `selected.csv` changes just tokens **1024 and 2048**. The original
three-row choice is preserved as `screen_selected.csv` and
`screen_selection.json`. Because main later faulted too, 8192's rejection is
provisional: re-evaluate it under controlled GPU access before concluding its
configuration is bad.

For the two-row selected configuration, two complete fresh paired runs passed
all 15 cases. Raw native times below are internal handoff evidence; the final
user deliverable should be TFLOPS/BW.

| Pair | Tokens | Main G1/G2 us | Selected G1/G2 us |
|---:|---:|---:|---:|
| 1 | 1024 | 82.7698 / 46.7469 | 80.0159 / 44.5954 |
| 1 | 2048 | 86.3995 / 59.4607 | 84.6849 / 57.2616 |
| 2 | 1024 | 83.9740 / 47.3335 | 79.6207 / 43.9809 |
| 2 | 2048 | 87.2110 / 59.6512 | 85.0487 / 57.0360 |

Pair order was main->selected, then selected->main, then main->selected.
The third main retry completed all 15 cases, but the third selected run failed
at 8192, even though its 8192 CSV row was now identical to main. Thus **there
are not three complete successful final pairs**.

## Fault chronology: do not misattribute

All logs are in the artifact root. Faults were GPU memory-access faults with
"Page not present or supervisor privilege", not correctness tolerance failures.

1. `verify_r2_selected.log`: failed at token=8192 with the original copied
   N512 candidate. Earlier screening and verification had passed that case.
2. `fault_check_selected.log`: isolated token=8192 N512 candidate faulted again.
3. `fault_check_main.log`: isolated original main token=8192 passed.
4. `finalverify_r3_main.log`: **unmodified main** faulted at token=2048.
5. `finalverify_r3_main_retry1.log`: retry of full main passed all 15 cases.
6. `finalverify_r3_selected.log`: faulted at token=8192 with the **original main
   8192 row**, since that candidate had already been reverted. Tokens 1024 and
   2048 had passed earlier in that same process.

Items 4 and 6 rule out claiming the fault is demonstrated to be unique to the
copied N512 configuration. Root cause has not been diagnosed. User suggests
concurrent processes. Preserve all failed logs, and do not quietly omit this
limitation from the final conclusions if it remains unresolved.

No kernel code, driver settings, clocks, or host NUMA settings were changed.

## Scripts and artifacts

Everything below lives in `/workspace/trace/kimi3_tp8_20261119/` and is copied
into the committed `tune_kimi_artifacts/` snapshot:

- `main.csv`: exact TP8 CSV from main baseline.
- `donor.csv`: previous DSV branch's non-EP schedule source snapshot.
- `n256.csv`, `n512.csv`: full copied candidate tables; G1 N256 in both.
- `mapping.json`, `metadata.json`: donor mapping, revisions, hashes, scope.
- `screen_selected.csv`, `screen_selection.json`: original 3-row selection.
- `selected.csv`, `selection.json`: provisional 2-row selection, not applied.
- `tune.py`: prepare candidates and launch native benchmark variants; records
  commands, waits for lock/idle, checks correctness and exact CSV lookup.
- `idle_exec.py`: idle check inside optional lock; with no wrapper, polls for
  process idleness. With wrapper, exits 75 on busy so the caller releases it.
- `analyze.py --screen`: constructs initial selection from screening results.
  **Running it again overwrites selected.csv with the original three-row
  screening choice**, so preserve any subsequent decision first.
- `verify.py`: three full paired runs of main/selected, using names
  `finalverify_r{1,2,3}_{main,selected}`. Re-running it overwrites those logs;
  use a fresh tag/directory if preserving the current evidence.
- `finish_verification.py`: attempted third-pair retry. Main retry succeeded;
  selected failed at 8192. It must not be treated as complete.
- `report.py`: **UNEXECUTED, UNFINISHED DRAFT** which aggregates three complete
  pairs, applies selected rows to the production CSV, and writes a table.
  It currently expects missing `finalverify_r3_selected_results.json`.
  Its narrative about faults is stale/incomplete because more faults occurred
  after it was written. Update it before use. It mutates the production CSV;
  do not run it merely to inspect data or before validation is complete.
- `*_results.json`: successful full-run stage measurements and correctness.
- Matching `.json` launch manifests and `.log` native output.

`finalverify_r3_main_results.json` is a copy of the successful retry result;
its records point to `finalverify_r3_main_retry1` as the actual source log.
It does not represent a successful original third main attempt.

There is **no completed PERFORMANCE.md / PERFORMANCE.csv / final.csv** yet.
Do not assume the draft report has applied any production changes.

## Provisional two-row schedule changes relative to main

This explains `selected.csv`; it is not an instruction to apply it immediately.

Token 1024, tile_m=tile_m2=32:

- tile_n: 128 -> 256; tile_k: 256 -> 128.
- m_warp: 2 -> 1; n_warp: 2 -> 4; n_warp2: 2 -> 4.
- tdm_as_in_prologue: 0 -> 1; tdm_b_th: 0 -> 1.
- waves_per_tensor_tdm and waves_per_tensor_tdm2: blank -> -1.
- Stale recorded us: 174.5391 -> 0.

Token 2048, tile_m=tile_m2=64:

- tile_n2: 128 -> 512; tile_k2: 128 -> 256.
- m_warp: 2 -> 1; n_warp2: 2 -> 4.
- tdm_as_in_prologue: 0 -> 1; rolled_drain: 1 -> 0.
- waves_per_tensor_tdm and waves_per_tensor_tdm2: blank -> -1.
- Stale recorded us: 221.3278 -> 0.

All non-N changes above come from copying the one chosen donor schedule; they
were not independently swept. Model shape, activation, quantization, topk,
experts, and both M tiles stay unchanged.

## Resume steps and final acceptance

1. Read this document, the current skill files, and relevant repository rules.
   Inspect branch/worktree before changing anything; preserve unrelated files.
2. Establish the GPU-lock availability in the new environment and check for
   competing jobs. The user wants to finish this task, not request repeated
   permission for already authorized benchmarking and CSV edits.
3. Reuse candidate snapshots and successful results. Do not repeat a broad
   sweep or introduce extra knobs. With reliable isolation, reassess the
   failed 8192 candidate if useful, since its earlier +6.24% gain may be valid.
4. Complete a controlled, paired main-versus-selected verification. Preserve
   failures and repeat incomplete pairs rather than presenting partial output
   as a completed run. Recheck correctness for the final combined rows.
5. The draft criterion was >=1% combined throughput gain and positive gains
   in all three paired runs. This is a conservative measurement choice, not
   a user-mandated threshold. Any final decision needs evidence exceeding
   observed timing noise, and slower/no-gain rows should keep main.
6. Apply only validated selected rows to the dedicated TP8 production CSV.
   Preserve all other rows and all M tiles. Do not carry donor performance
   metadata into Kimi rows. Run git diff --check and verify only intended CSV
   rows/fields changed. No new kernel code or broad test suite is needed.
7. Deliver all 15 cases in **one table** with main and tuned GEMM1/GEMM2 TFLOPS
   and effective bandwidth, plus clear per-stage/combined gains. State which
   rows changed and retained N widths. Link raw results and the CSV.
8. The BW model is the existing CPU-pure `_gemm_work_metrics` function in the
   benchmark. TFLOPS = FLOPs / us / 1e6; TB/s = logical bytes / us / 1e6. It
   counts active experts and fused stage formats. This is **effective logical
   bandwidth, not measured physical HBM traffic**, and excludes communication
   and auxiliary kernels. Use the same model on both sides.
9. For unchanged rows, either show separate noisy measurements with an
   unchanged label, or pool identical-config measurements and explicitly say
   so. Do not manufacture gains from run-to-run differences in identical rows.
10. Do not claim completion or a confirmed improvement from the preliminary
    table alone. The user explicitly requested committing this work for handoff.
    This is a local work-in-progress commit; a push was not requested.

## Process state at handoff

A process-list check after the user's handoff request found **no running
benchmark, tune.py, idle_exec.py, verify.py, or finish_verification.py process
from this task**. The last runner exited after the selected third-pair fault.
No other user's process was killed. No task-owned waiting benchmark remains.

If moving to a different filesystem/container, transfer this branch including
the handoff commit. It contains this document and `tune_kimi_artifacts/` with
all raw logs and candidate CSVs. Preserve or recreate the pinned Python
environment separately. No GPU kernels, compiled caches, venv packages, or
Python bytecode are included in the commit.
