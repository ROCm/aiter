# S1F pull dispatch: current code and experiment handoff

**Latest refinement: persistent GEMM2 now rendezvous immediately before the next
task's first TDM issue, using raw signal/wait. Per-wave output tensor_wait(0)
remains at the previous task's end.** Two fresh interleaved runs per mode give
GEMM2 **561.66 -> 545.44us (-2.89%)** and native E2E **1577.90 -> 1526.04us/layer
(-3.29%)**, versus the same persistent256 configuration with the old task-end
barrier. Full-size correctness and emitted ISA checked; no new flag needed.
See [barrier relocation results](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_EP_PREISSUE_BARRIER_20261119.md).
The earlier comparisons below remain historical and use different controls.

**Latest measured selection (2026-11-19): original gather/preshuffled S1F,
P96/C160 with affinity3:5, and GEMM2 TDM gather scatter with persistent256.**
At16384 tokens/rank (65536 global), two interleaved runs per mode give
GEMM2 **591.41 -> 548.64 us** and native E2E/layer **1590.27 -> 1533.50 us**
versus the same workspace's default GEMM2 grid. Full-size changing-routing
correctness passed. This is not a comparison against a fresh origin/main.
See [paired results and reproduction](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_EP_TDM_PERSIST_20261119.md).

```bash
AITER_MOE_EP_BUFFER_STORE=0 AITER_FLYDSL_PERSISTENT_WORKERS2=256 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
```

The source-shuffle and buffer-store experiments below remain optional; this
selected configuration uses neither. Older sections retain experiment history.

**当前实现的关键保证：开启 `AITER_MEGA_PULL_SRC_SHUFFLE=1` 后，每个 M256
内的所有有效行都来自同一个 source rank。Padding 行不对应 token，反向路由为
`(-1, 0)`。这个保证适用于 normal routing，不依赖 balanced routing。**

**Status (2026-11-19): wave-level ordinary TDM in GEMM1 and contiguous TDM
writeback in GEMM2 both pass EP4 correctness at 257 and 16384 tokens/rank.**
Each size runs six changing-input graph replays, including random routing,
hot/empty experts, disjoint ownership and unequal source tails. The current
priority is GEMM1/GEMM2 latency; planner and E2E overhead are recorded separately
and are not acceptance criteria for this experiment.

This branch also retains the earlier optional scheduling experiments, M256-ready
support and their tests. Their history is in
[the kernel README](aiter/ops/flydsl/kernels/mega_moe_gfx1250/README.md).
The new packing/source-shuffle switches default to off. CK is not part of this
change. This checkpoint was prepared on `dev/yadai_dispatch_pull`; use
`git rev-parse HEAD` to record the exact revision on the next machine.

## Workload and immediate objective

DSV4.1 Flash, EP4: **16384 tokens/rank =65536 global tokens**,384 experts,
96 local experts,topk6,H5120,I2304,MXFP4,8 layers. Use the explicit
`aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv`, with EP `topk=-1` lookup. The workload key is65536; inspect the effective
configuration rather than assuming an exact CSV row exists. Do not confuse
this with non-EP token16384.
GEMM1 is256x256x256,2x2 waves,4 buffers;256 workers,P96/C160,affinity3:5,
legacy instruction prefetch. GEMM2 stays256x256x256,2x2 waves,4 buffers,WPT1,cluster1x4.
`AITER_MEGA_EP_LINEAR_STORE=1` changes only its writeback layout and combine
slot addressing; its GEMM mainloop remains unchanged.

**Latest user priority: get GEMM1 and GEMM2 measured first.** Source packing
and contiguous landing are allowed even if compact-plan/E2E become slower.
The original gather path remains the default and the comparison baseline.

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
only its own tail's inverse mappings to route=-1,weight=0. Existing gather-store GEMM2 drops negative routes. The optional linear-store
path writes padded rows to separately reserved padding slots; the final reduce
never reads them.

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
- With source shuffle, each producer wave loads four M16 frames (M64) in one
  ordinary 2D TDM per K segment. Default1024 payload bytes/row gives three K
  segments for H5120. There is one tensor wait per M64 K segment, not per M16.
  Payload is already preshuffled; no gather or producer shuffle is required.
  `AITER_MEGA_PULL_WAVE_LOAD=0` retains the old per-M16 packed copy.
- Store completion and the existing A-ready protocol precede consumption.
  Reproduction scripts use the old M64 flags to isolate the layout experiment.
  Consumer scheduling, task prefetch and GEMM code are retained.

### Frame format

```text
source buffer [destination rank][destination M16 frame]:
  payload [K/32][16 rows][16 bytes]   -- already A-preshuffled
  scales  [16 rows][K/32 bytes]      -- row-major scales
```

For H5120 a frame is16*(2560+160)=43520 bytes. A wave's full M64 wire
is174080B; four such waves exceed320KiB LDS. Default K payload1024B uses
three 2D loads per wave: `(4,16384B)`, `(4,16384B)`, `(4,10752B)`, all with
43520B frame stride. The final load includes scales. Four waves use256KiB
scratch; for nb4 planar GEMM, a64KiB hole separates waves1/2 so the existing
GEMM control area remains untouched inside the total320KiB allocation. Scale
reads use the same shifted wave slice as payload reads. K bytes are tunable
with `AITER_MEGA_PULL_LOAD_K_BYTES` (positive multiple of16; LDS limits checked).
Packed offsets use64-bit global pointers because source storage can exceed4GiB.

Allocation is `world * ceil(compact_cap/16) * frame_bytes`, plus small completion
storage. It is a worst-case sparse allocation, not a storage-optimal design.
Packed load without source shuffle retains a mixed-source M16 fallback: load each
present source frame and store only rows owned by it. With source shuffle that
fallback should not be needed for live frames; the current code still retains it.

## Optional GEMM2 ordinary TDM writeback

Enable `AITER_MEGA_EP_LINEAR_STORE=1` with packed/source-shuffled S1F pull.
The first version requires bf16 combine and experts/rank divisible by32.
Source-aligned M256 alone is insufficient: original `(token, topk slot)`
destinations are still noncontiguous. The existing compact-plan additionally:

1. Assigns a padded home landing range to each source's global expert segment.
   Capacity per source is `align256(max_routes + global_experts*255)` rows.
2. Publishes one destination-rank/landing-base entry per destination M256.
3. Writes a local original-route-to-home-slot map for the final combine.

GEMM2 retains original weights and rowmap, GEMM tiling, mainloop, and LDS store
layout. Each wave sends its entire M64 N tile using **one ordinary 2D TDM store**
instead of eight gather-stores. The tensor extent excludes the16B LDS padding,
so adjacent N tiles do not overwrite each other. Existing per-wave tensor waits
and cross-rank combine barrier remain. Combine reads the mapped slots and sums
in the same top-k order. No extra GPU kernel is introduced for these mappings.

Default is off. This is a GEMM2 latency experiment, not a claim of faster E2E.

## Code map

| File | Relevant responsibility |
|---|---|
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py` | `compile_tdm_compact_plan`: source-aligned prefix, padding maps, live metadata, fused quant/packing and local map generations |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/packed_wave_copy.py` | M64 wave-level ordinary TDM and K-segment LDS geometry |
| `aiter/ops/flydsl/kernels/mega_moe_gfx1250/linear_combine.py` | Feature selection and padded home-slot bound |
| `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` | Optional ordinary TDM GEMM2 writeback |
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

GEMM1 comparison, keeping GEMM2 unchanged unless explicitly enabled (unique run labels required):

```bash
BENCH_PLAN_PACK=0 BENCH_SRC_SHUFFLE=0 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_baseline_01
BENCH_PLAN_PACK=1 BENCH_SRC_SHUFFLE=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_src_shuffle_01
```

Both GEMMs use ordinary TDM transfers (unique label required):

```bash
BENCH_PLAN_PACK=1 BENCH_SRC_SHUFFLE=1 BENCH_EP_LINEAR_STORE=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_src_linear_01
```

For correctness, add `BENCH_EP_LINEAR_STORE=1` to the `check_src*` commands.
For native timing only, use `AITER_MEGA_EP_LINEAR_STORE=1 bash run_ds41_flash_src_shuffle.sh`.

Add `BENCH_BALANCED=false` to both for normal routing. Optionally isolate packing
without M256 source padding using `BENCH_PLAN_PACK=1 BENCH_SRC_SHUFFLE=0`.
Native E2E is host wall time for an8-layer graph replay plus final synchronization;
divide by8 for per-layer time. Stage times use explicit HIP graph event nodes in
a subsequent pass, with event overhead. Do not sum stage medians to reconstruct
native E2E. Confirm logs show `packed_input=True`, `src_rank_shuffle=True`,
`tile_m=256`, `ready_first=False`, `producer_group=1`,P96/C160. Planner symbol
contains `_packed_a` and `_src256`; ready symbol contains `_src256`.

## Measured GEMM1 / GEMM2 result

Two runs per configuration, balanced16384/rank, EP4. Values are the arithmetic
mean of per-run HIP-event medians, then averaged equally across the four GPUs.
Full per-GPU results, individual runs and symbols are in
[RESULTS_20261119.md](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_20261119.md)
and [results_20261119.json](op_tests/multigpu_tests/mega_moe_diagnostics/results_20261119.json).

| Configuration | GEMM1 us | GEMM2 us |
|---|---:|---:|
| Original gather baseline |545.55|591.92|
| GEMM1 ordinary TDM, GEMM2 gather-store |533.12|592.61|
| GEMM1 ordinary TDM + GEMM2 ordinary TDM store |541.17|587.89|

Both-TDM gains are only0.80%/0.68% across GPUs; GPU2 gains are0.94%/1.40%.
The larger initial GEMM1-only gain did not fully repeat. Fewer TDM instructions
have not produced a large measured gain. Keep the experiments optional.

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

## Validation notes and remaining experiments

- Initial post-reboot source-shuffle validation found a null `tile_storage`
  pointer in the external-ready planner arguments. It is now passed when source
  shuffle is enabled, and the existing test checks it before launch. Retry and
  full-size tests passed. Boot IDs are recorded in each diagnostic directory.
- Wave-load correctness passed at257 and16384/rank. The full-size check covers
  the nb4 LDS control gap. Linear GEMM2 passed both sizes with six changing-input
  replays and original push reference. Ordinary-TDM timing must show `_eplinear`
  in GEMM2's symbol, not `_epscatter`.
- Normal routing can increase GEMM work because every source segment rounds up
  to M256. Balanced timings must not be presented as normal-routing timings.
- Prior fine-cycle data above are historical. A second uncontended fine-cycle
  capture remains a separate experiment; current priority is GEMM1/GEMM2 timing.
- Next-machine commands need the installed AITER/MORI and pinned FlyDSL0.5.0.
  No temporary overlay under `/workspace/gpubench` is needed for the TDM paths.
  Keep source readiness, A readiness, and combine completion distinct.

## Unique pull with local row-major scatter (experimental)

`AITER_MEGA_PULL_UNIQUE=1` replaces the S1F pull producer. Each wave owns
16 source tokens, fetches each needed payload/scale once for this destination
rank, and fans out from LDS to all matching local expert rows. This is a
hybrid: remote pull is deduplicated, local writes still occur once per route.
The source wire stays in its original per-token layout; compact-plan writes
its final route map into peer-visible storage. There is no separate unique
cache or extra local gather kernel.

The local expert payload is **row-major**. Each wave issues two indexed TDM
stores per top-k slot for its16 loaded rows, reusing LDS directly. The output
row width is the payload width; TDM drops scale/padding columns and masked
nonlocal destinations. Small scale rows use contiguous lane stores while the
payload stores are in flight. GEMM1 explicitly
uses `a_preshuffled=0`; the existing GEMM1 quant epilogue and GEMM2 entry point
also use matching row-major payloads. This differs from the source-shuffle
experiment above: destination M256 tiles are not guaranteed source-homogeneous,
and `PLAN_PACK`, `SRC_SHUFFLE` and `EP_LINEAR_STORE` must remain off.

The original external-ready kernel waits for source quantization and final
route-map visibility. It also resets local M64 completion counters. After
fan-out stores retire, each source row contributes one AcqRel agent atomic per
local route; the last writer publishes the original M64 A-ready generation.
Consumers and producer-to-consumer transition retain their original scheduler.
No new cross-device completion atomic is required.

```bash
# Six changing-input correctness replays, including hot/empty experts and tails.
CHECK_TPR=257 CHECK_UNIQUE=1 BENCH_UNIQUE_PULL=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_unique_small_NEW
CHECK_TPR=16384 CHECK_UNIQUE=1 BENCH_UNIQUE_PULL=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_unique_flash_NEW
# Explicit HIP event stage timing and a separate native E2E pass.
BENCH_UNIQUE_PULL=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_unique_NEW
BENCH_UNIQUE_PULL=0 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_unique_baseline_NEW
```

Use unique labels for every run. Add `BENCH_BALANCED=false` to both timing
commands for random top-k routing. The balanced pattern has six routes to one
destination per token (4.5 remote routes but0.75 remote destinations/token).
That sixfold reduction in requested peer payload bytes does not apply to
random routing. Local output byte volume is unchanged; latency must be measured.

Final row-major/TDM correctness passed at257 and16384 tokens/rank. This version
is slower than the original pull, so `AITER_MEGA_PULL_UNIQUE` defaults to0.
Balanced GEMM1/GEMM2 are894.38/580.57us versus538.56/579.77us for the baseline;
random routing is1503.27/578.20us versus595.03/585.23us. One successful run per
configuration, averaging the four GPU HIP-event medians. GEMM1 includes dispatch.
Full results, limitations and the intermediate layout experiments are in
[RESULTS_UNIQUE_20261119.md](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_UNIQUE_20261119.md)
and [results_unique_20261119.json](op_tests/multigpu_tests/mega_moe_diagnostics/results_unique_20261119.json).

Native timing only, with the same hybrid P96/C160 and affinity:

```bash
AITER_MEGA_PULL_UNIQUE=1 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
```

The script name reflects the original default, not a claim that the optional
hybrid is faster. Source-map scans, per-route completion RMWs and changed
consumer startup remain candidates for further cycle-level analysis. The next
optimization must address destination-tile completion as well as copy layout.

## Persistent GEMM2 remote buffer scatter (experimental)

`AITER_MOE_EP_BUFFER_STORE=1 AITER_FLYDSL_PERSISTENT_WORKERS2=256` changes only
GEMM2's unquantized EP scatter. Start from the original gather/preshuffled S1F
baseline, with unique pull, source packing and linear GEMM2 stores disabled.

Each compute wave weights its accumulator in fp32, converts to bf16, and
stages its output in the existing LDS layout. For M divisible by16*num_waves,
output ownership becomes full-N row stripes: at M256/N256/waves2x2 each wave
writes64 rows of512B using `buffer_store_b128`. A raw LDS barrier precedes
cross-wave reads. Smaller M retains compute-wave ownership and64b stores.
Row addresses are precomputed lane-wise; four descriptor/data pairs are kept
live together to reduce premature resource reuse. Wave-uniform row descriptors
select the source rank's combine landing slot; invalid rows/columns are masked.
Output uses no TDM stores, so the next task's tensor counter covers input only.

All workgroup barriers inside this mode use `s_wait_dscnt 0` followed by
`rocdl.s_barrier_signal/wait`. A single `s_wait_storecnt 0` follows the outer
persistent loop, immediately before kernel exit. The original stream ordering
and combine cross-rank barrier retain remote completion guarantees.
`xcnt` waits remain where LLVM needs XNACK address-translation operand
protection; they do not represent remote-store completion. They must not be
removed solely to reduce the instruction count.

```bash
# Native timing; preserve original gather GEMM1 and affinity3:5.
AITER_MOE_EP_BUFFER_STORE=1 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
# HIP-event breakdown; use fresh labels, flags default to the original path.
BENCH_EP_BUFFER_STORE=1 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_buffer_NEW
BENCH_EP_BUFFER_STORE=0 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_tdm_persist_NEW
BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_original_NEW
# Changing-routing correctness and actual ISA (repeat with CHECK_TPR=16384).
CHECK_TPR=257 CHECK_EP_BUFFER=1 BENCH_EP_BUFFER_STORE=1 BENCH_GEMM2_WORKERS=256 BENCH_DUMP_ASM=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_ep_buffer_NEW
```

`audit_ep_buffer_asm.py PATH_TO_22_final_isa.s` checks the emitted CFG: no
storecnt wait can belong to a control-flow cycle, the sole store wait must be
in the exit block, and output must contain buffer stores without tensor stores.
The runtime symbol contains `_pw256_` and `_epbuffer_fullrow4_` for the Flash
shape (or `_epbuffer_group4_` for smaller M); A remains preshuffled.
The mode rejects quantized combine, linear landing maps, split-K, nonpersistent
launches, and per-peer landing extents reaching4GiB.

**This option is slower and remains disabled by default.** The retained full-row4
version measured GEMM2/E2E **669.89/1650.96 us** in one run. Earlier compute-wave
rows measured729.77/1685.21 us. The full-row version has64 static buffer_store_b128
instructions, no tensor stores and one exit-only storecnt. Small/full changing
routing checks passed. A non-temporal variant passed full correctness but was
not timed; it is saved in `ep_buffer_nt_experiment.patch`, not enabled in source.
XCNT also drains implicitly at branches/raw barriers, so explicit wait counts
alone cannot establish latency. See [buffer experiment history](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_EP_BUFFER_20261119.md)
and [selected TDM comparison](op_tests/multigpu_tests/mega_moe_diagnostics/RESULTS_EP_TDM_PERSIST_20261119.md).
