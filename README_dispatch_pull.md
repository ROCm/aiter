# S1F pull dispatch: current code and experiment handoff

**当前实现的关键保证：开启 `AITER_MEGA_PULL_SRC_SHUFFLE=1` 后，每个 M256
内的所有有效行都来自同一个 source rank。Padding 行不对应 token，反向路由为
`(-1, 0)`。这个保证适用于 normal routing，不依赖 balanced routing。**

**Status: source-rank shuffle is implemented but has not passed GPU validation
or a performance comparison yet.** GPU access was occupied when this checkpoint
was requested. Do not treat the invariant above as a completed test result.
The earlier packed-load prototype, without source-segment padding, passed EP4
correctness at257 and16384 tokens/rank. It is not evidence that the new padding
implementation is already correct or faster.

This branch also retains the earlier optional scheduling experiments, M256-ready
support and their tests. Their history is in
[the kernel README](aiter/ops/flydsl/kernels/mega_moe_gfx1250/README.md).
The new packing/source-shuffle switches default to off. CK is not part of this
change. This checkpoint was prepared on `dev/yadai_dispatch_pull`; use
`git rev-parse HEAD` to record the exact revision on the next machine.

## Workload and immediate objective

DSV4.1 Flash, EP4: **16384 tokens/rank =65536 global tokens**,384 experts,
96 local experts,topk6,H5120,I2304,MXFP4,8 layers. Use the explicit
`aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv`, whose EP row
has `topk=-1` and token key65536. Do not confuse this with non-EP token16384.
GEMM1 is256x256x256,2x2 waves,4 buffers;256 workers,P96/C160,affinity3:5,
legacy instruction prefetch. GEMM2 settings and implementation are unchanged.

**User priority: measure GEMM1 first. Only try a corresponding GEMM2 optimization
if GEMM1 shows a clear performance advantage.** Also report compact-plan/quant
and native E2E: making GEMM1 faster by moving more work into the planner may
still lose overall. Preserve the fastest existing gather path as the baseline.

## What source-rank shuffle means

The old compact plan already makes `(destination, expert, source_rank)` segments,
but source counts are not generally multiples of256. Consequently a tile can
cross a source boundary. In a saved balanced16384/rank run, every rank had384
full M256 tiles; all were single-source, but **none** had consecutive source-token
addresses. Same-rank ownership does not imply a contiguous source tensor.

The new mode has two distinct layout operations:

1. **Source-rank row grouping/alignment:** for each expert, assign each source
   `ceil(actual_count /256)*256` rows. Source starts are M256 aligned. All valid
   rows in any M256 therefore share one source rank, even with random routing.
   Empty segments reserve zero rows. A partial segment has padding at its tail.
2. **FP4 source packing and A preshuffle:** quantize a source token/group once,
   then write its quantized bytes/scales into each topk route's final destination
   frame. This makes source storage actually contiguous for ordinary TDM loads.
   Row grouping alone cannot replace gather when source token indices differ.

This first implementation pays extra padding/computation under normal routing.
`masked_m` and `psum` describe **padded compute extents**. Per-M64 `tile_info`
describes **actual live rows**, including zero-live pieces inside a padded M256.
The producer never treats a padding row as a source token. Every source clears
only its own tail's inverse mappings to route=-1,weight=0. Existing GEMM2 scatter
checks route>=0 and drops padded outputs. No GEMM2 optimization is included.

## Current execution and synchronization

- Existing compact-plan exchanges raw histogram counts. The source prefix scan
  sums rounded per-source extents in source-shuffle mode; histogram counts stay
  unrounded for live-row metadata.
- Plan block0 builds per-M64 metadata from real source counts and padded starts.
  Capacity and per-step row bounds reserve `world_size * experts_per_rank` tails.
  The chosen source-group count is frozen in geometry.
- Planning blocks write final source-route to destination-row maps and inverse
  maps, finish their stores and release-publish local completion generations.
- Quant TGs compute their first group while planning proceeds, then wait for the
  final map before scattering packed output. This introduces a map dependency:
  it does **not** retain all of the old plan/quant overlap. Quant waves maintain
  their own generation counters, incremented once per plan invocation, including
  waves with no token work. Do not replace these with a late read of barrier[2];
  block0 may already have incremented it. Plan+quant TG count is limited to CUs.
- The existing external-ready kernel waits for all source plan/quant completion.
  In source-shuffle mode it preserves the planner's live-row metadata rather
  than rebuilding dense expert metadata. Histogram completion alone is not
  sufficient for a remote packed-data read.
- Each producer wave loads an M16 frame with **ordinary `tdm_load`**, waits for
  that load, and copies the already-shuffled payload and row-major scales out.
  The current producer still processes four M16 pieces per M64; this checkpoint
  does not claim a new TDM pipeline. Four waves cover an M256.
- Store completion and the existing A-ready protocol precede consumption.
  Reproduction scripts use the old M64 flags to isolate the layout experiment.
  Consumer scheduling, task prefetch and GEMM code are retained.

### Frame format

```text
source buffer [destination rank][destination M16 frame]:
  payload [K/32][16 rows][16 bytes]   -- already A-preshuffled
  scales  [16 rows][K/32 bytes]      -- row-major scales
```

For H5120 a frame is16*(2560+160)=43520 bytes. Producer issues one43520B ordinary
TDM load into its existing per-wave LDS slice. Packed offsets use64-bit global
pointers because the sparse source allocation can exceed4GiB.

Allocation is `world * ceil(compact_cap/16) * frame_bytes`, plus small completion
storage. It is a worst-case sparse allocation, not a storage-optimal design.
Packed load without source shuffle retains a mixed-source M16 fallback: load each
present source frame and store only rows owned by it. With source shuffle that
fallback should not be needed for live frames; the current code still retains it.

## Code map

| File | Relevant responsibility |
|---|---|
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py` | `compile_tdm_compact_plan`: source-aligned prefix, padding maps, live metadata, fused quant/packing and local map generations |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/packed_copy.py` | `make_packed_copy`: ordinary TDM frame load and contiguous shuffled-A stores |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/stages/pull.py` | Switch validation, packed regions, completion initialization, compile options/cache keys |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/stages/dispatch_compact.py` | Source-group capacity and per-step row bounds |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/s1f_pull.py` | Preserves planner metadata in external-ready mode; selects packed producer |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/arena.py` | Registers packed source and completion regions |
| `op_tests/multigpu_tests/test_mega_moe_gfx1250.py` | `_check_pull_s1f_src_rank_shuffle`: arbitrary routing and padding correctness |
| `op_tests/multigpu_tests/mega_moe_diagnostics/` | Targeted checks, native E2E/HIP-event timing and fine-cycle overlay generator/analyzer |

## Reproduce on another machine

Use gfx1250 EP4 and the pinned FlyDSL version (this work used stock0.5.0).
Install AITER/MORI as usual. Scripts use `/workspace/mori-plan-describe` only if
present; set `DS41_MORI_ROOT` for another source build, otherwise installed MORI
is used. Commands below are from the repository root. Scripts check
`rocm-smi --showpids`, poll every second while busy, and use `gpu-lock` if present.
Do not overlap another user's GPU work. The diagnostic runner limits GPU jobs
to240 seconds and records command, env, source diff and boot IDs.

First run small correctness, then full Flash correctness. These tests keep push
and pull arenas alive together, so the targeted runner reserves16GiB VMM.
They compare outputs, exact FP4 bytes/scales/routing weights, exact route sets,
M256 source uniqueness, live metadata and six changing-input graph replays:
random routing, hot/empty experts, disjoint ownership and unequal source tails.

```bash
CHECK_TPR=257 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_small_01
CHECK_TPR=16384 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_flash_01
```

Native E2E for balanced or normal routing:

```bash
bash run_ds41_flash_src_shuffle.sh
DS41_BALANCED=false bash run_ds41_flash_src_shuffle.sh
```

GEMM1 comparison, keeping GEMM2 unchanged (unique run labels required):

```bash
BENCH_PLAN_PACK=0 BENCH_SRC_SHUFFLE=0 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_baseline_01
BENCH_PLAN_PACK=1 BENCH_SRC_SHUFFLE=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_src_shuffle_01
```

Add `BENCH_BALANCED=false` to both for normal routing. Optionally isolate packing
without M256 source padding using `BENCH_PLAN_PACK=1 BENCH_SRC_SHUFFLE=0`.
Native E2E is host wall time for an8-layer graph replay plus final synchronization;
divide by8 for per-layer time. Stage times use explicit HIP graph event nodes in
a subsequent pass, with event overhead. Do not sum stage medians to reconstruct
native E2E. Confirm logs show `packed_input=True`, `src_rank_shuffle=True`,
`tile_m=256`, `ready_first=False`, `producer_group=1`,P96/C160. Planner symbol
contains `_packed_a` and `_src256`; ready symbol contains `_src256`.

## Fine-cycle findings and limitations

Fine probes are an opt-in generated overlay of the current gather implementation;
they do not add clocks to production by default. Only subtract timestamps within
the same wave. Stores added by probes can perturb later waits. These are medians
from one captured final layer/replay, **not yet verified by a second capture**:

| Producer interval, cycles | Local source | Remote source |
|---|---:|---:|
| M64 copy |25700|67899|
| M64 final store wait |828|663|
| M64 flag issue |99|99|
| First M16 gather/descriptor issue |866|983|
| First M16 tensor wait |1954|10500|
| First M16 payload LDS/shuffle/store issue |2750|2932|

Consumer wave0: whole-task median22567 cycles; first-task A-ready31050;
later-task A-ready34; queue-claim issue43. The current evidence points to remote
TDM completion and startup waiting, not the queue-claim instruction itself.
Do not multiply M64 times by4 to infer M256 latency; four waves run concurrently.

Instrumented native E2E median1583.82us/layer had severe outliers(mean3176.13).
A fresh uninstrumented run had median1586.88,mean1588.35us/layer. This is not an
instrumentation speedup. Re-capture uncontended cycles before trusting fine deltas:

```bash
BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity cycles_repeat_01
python op_tests/multigpu_tests/mega_moe_diagnostics/analyze.py op_tests/multigpu_tests/mega_moe_diagnostics/ipflegacy_p96_cycles_repeat_01
```

## Next actions / do not lose these constraints

1. Validate the committed source-rank padding code first; fix issues before timing.
   Earlier packed-load correctness does not cover this added padding logic.
2. Compare baseline/packed/source-shuffled GEMM1 and compact-plan with identical
   settings. Record native E2E and normal-routing padding overhead.
3. If GEMM1 gains, then investigate applying the same source-rank property to
   GEMM2/combine. Do not assume same-rank destinations imply contiguous token slots.
4. Keep source readiness and A readiness distinct. Keep negative padding mappings
   and live metadata correct across graph generations and empty experts.
5. Repeat the fine-cycle capture. A packed benchmark was stopped when foreign
   processes appeared during compilation; no valid packed performance result
   exists at this checkpoint. Do not recycle that run as a benchmark.

Original raw artifacts are local to the previous host:
`/workspace/gpubench/s1f_fine_cycles_20261119`,
`/workspace/gpubench/s1f_plan_pack_20261119`, and
`/workspace/gpubench/s1f_src_rank_shuffle_20261119`.
They are not required to run the committed scripts and are not presumed present
on another machine. No cluster, queue or GEMM2 experiment should be silently
combined with this comparison.
