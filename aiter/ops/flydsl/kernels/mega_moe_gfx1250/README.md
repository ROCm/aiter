# MegaMoE FP4 pull dispatch on gfx1250

## Reproduce the best measured DSV4.1 Flash configuration

From the repository root:

```bash
bash run_ds41_flash_best.sh
```

The script runs the existing native MegaMoE benchmark on four gfx1250 GPUs:
16384 tokens per rank (65536 global), 384 experts, top-k 6, H5120/I2304,
balanced routing, eight layers, five warmups and 32 iterations. It explicitly
loads `aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv`, enables
S1F pull with A preshuffle, uses 64 producers, and enables fused GEMM2. The
selected GEMM1 cluster falls back to 1x1; GEMM2 retains its CSV cluster 1x4.
It clears old tuning overrides and checks GPU idleness every second before
launching. FlyDSL must match `requirements.txt` (0.5.0 at this revision).

On the development host it uses the MORI build in
`/workspace/mori-plan-describe` when present. Set `DS41_MORI_ROOT` to select
another compatible MORI checkout; otherwise the installed MORI is used.
The benchmark prints native E2E timing; per-kernel profiling is disabled.
The GEMM breakdown below came from the separate HIP-event harness.

The committed producer is the full-row gather version from the two successful
CSV-fed runs, not either K-sliced pipeline experiment. At the shape above,
latencies in microseconds (four-rank averages) were:

| Version | Plan+quant | Ready | GEMM1 incl producer | GEMM2+scatter | E2E/layer |
|---|---:|---:|---:|---:|---:|
| Push control in the same merged workspace | 36.10 | — | 625.54 | 597.04 | 1648.78 |
| **Committed full-row pull, two-run mean** | **39.96** | **12.65** | **570.04** | **595.48** | **1609.77** |
| Experimental K1024/M32/2 buffers, one run | 42.02 | 12.74 | 612.45 | 592.61 | 1641.65 |
| Experimental K1536/M16/3 buffers, one run | 38.27 | 12.65 | 630.76 | 598.67 | 1666.15 |

These are saved measurements, not a fresh benchmark at commit time. The push
control is not a separate clean-main checkout. E2E and kernel intervals are
measured separately; the listed stages do not sum to E2E. The K-sliced variants
passed Flash correctness but regressed E2E by 1.98% and 3.50% respectively.
Their source and patches are preserved on the development host under
`/workspace/gpubench/s1f_pull_k_pipeline_20261118/`; they are not enabled or
included in this commit. The restored kernel sources match the saved best-run
snapshot. Its H5120/I2304/tpr16384 and H7168/I1024/tpr17 correctness checks
passed, including changing graph inputs, empty ranks and exact dispatched
payload, scale and routing-weight comparisons.


This directory contains MegaMoE's expert-parallel dispatch, planning, and
combine kernels. The experimental pull path targets non-S1F prefill and writes
GEMM1's preshuffled activation layout directly. It is opt-in; ordinary push
dispatch remains the default.

## Enable pull

Set these variables before importing AITER:

```bash
export AITER_MEGA_DISPATCH_PULL=1
export AITER_MEGA_PULL_TDM_GATHER=1
export AITER_MEGA_S1F=0
export AITER_MEGA_QUANT_FUSE=128
export AITER_MEGA_PLAN_PREFETCH=0
```

The recipe must use compact dispatch with an FP4 wire and fused-scatter
combine. Pull requires positive `AITER_MEGA_QUANT_FUSE` and the supported
preshuffle dimensions. It does not replace S1F: if S1F is selected for a step,
that step keeps the existing S1F path. Explicitly disabling S1F makes a pull
comparison unambiguous.

`AITER_MEGA_PULL_TDM_GATHER=1` selects the 128-thread non-persistent kernel.
Without it, pull uses the earlier 512-thread persistent async-copy kernel.
The TDM kernel's symbol ends in `_tdmgather_t128_g2x8_grid`.

## Data flow

1. `stages/dispatch_compact.py` selects `stages/pull.py` for a non-S1F step.
2. The existing `compact_plan.py` launch runs routing workgroups alongside
   quantization workgroups. Quantization processes each local input token once
   and writes FP4 payload and E8M0 scales into symmetric source storage.
   Planning exchanges histograms and writes receiver-side `ep_rowmap` entries:
   packed source rank/route plus the routing weight bits. The packed route
   identifies both the source token and its top-k slot.
3. A stream event makes local dispatch wait for the entire local plan/quant
   launch. For TDM pull, a single-workgroup ready kernel publishes the epoch
   with system release ordering and acquires every peer's epoch. The copy
   kernel follows on the same stream. Histogram completion alone is
   insufficient. Both push and pull require data readiness.
4. `dispatch_pull.py` gathers remote source rows into private per-wave LDS,
   then stores preshuffled payload and row-major scales into local output.
5. `ExpertGrouping` carries `a_preshuffled=True`, `psum`, `masked_m`, and the
   inverse route map to GEMM1 and fused GEMM2/scatter. The GEMM input contract
   is `a_preshuffled=1`, `row_major_ascale=1`, `wire_row_stride=0`.
6. The existing combine barrier protects source storage from reuse by the
   next layer while peers still read it.

Quantization overlaps the histogram/routing work within the plan launch.
Pull waits for that whole launch; it does not consume partially quantized
source rows. Ordinary next-layer plan prefetch is suppressed for pull steps
because it cannot supply their quantized inputs.

## TDM geometry and layouts

For DSV4.1 Flash, `H=5120`, `I=2304`, EP4, 384 total experts, and top-k 6:

| Item | Value |
|---|---|
| Source payload | 2560 bytes/token, packed FP4 |
| Source scales | 160 bytes/token, one E8M0 byte per 32 values |
| Wire row | 2816 bytes: payload + scales + 96 bytes padding |
| Workgroup | 128 threads, four wave32s |
| Work per wave | One M16 tile; two eight-row TDM gathers issued before waiting |
| Work per workgroup | At most 64 destination rows, with no persistent tile loop |
| Grid | `max(1, ceil(row_capacity / 64))` workgroups |
| LDS per workgroup | `4 * 16 * 2816 + 96 * 4 = 180608` bytes |
| Output payload | `[M/16, K/32, M%16, 16 packed bytes]` |
| Output scales | Row-major `[M, K/32]` |

At 16384 tokens/rank and alignment 256, the existing conservative capacity is
417792 rows, so the grid contains 6528 workgroups. Uniform routing produces
about `16384 * 6 = 98304` live routes per rank, before expert padding.
Device-side `psum` skips unused tiles and empty experts. Grid sizing does not
assume balanced routing and does not read GPU counters back to the host.
More grid workgroups do not by themselves increase simultaneous residency;
the LDS footprint still matters.

Each wave owns its LDS slice and output vectors, eliminating per-tile
workgroup barriers. TDM readiness runs once before the copy grid; the copy
kernel retains its metadata barrier. The legacy async path retains per-TG
readiness. Payload
writeback uses 16-byte LDS loads and global stores; scales use four-byte
loads/stores. The local and legacy async paths reuse the extracted
`make_apre_payload_ops` helper in `../moe_apreshuffled_quant_scatter.py`.
The TDM path has its own per-wave writeback to avoid a 512-thread shuffle
barrier.

## Reproduce the prefill workload

Use the repository-pinned FlyDSL (0.5.0 at the development baseline) and a
compatible MORI installation exposing `mori.cco`. On this development host,
the tested MORI checkout is `/workspace/mori-plan-describe`; selecting that
checkout is a host-specific setup, not a portable installation requirement.

From the repository root, after selecting four idle gfx1250 GPUs:

```bash
PYTHONUNBUFFERED=1 ENABLE_CK=0 \
MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 \
MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27 \
AITER_MEGA_DISPATCH_PULL=1 AITER_MEGA_PULL_TDM_GATHER=1 \
AITER_MEGA_S1F=0 AITER_MEGA_QUANT_FUSE=128 AITER_MEGA_PLAN_PREFETCH=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_CONFIG_GROUPED_FMOE=aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv \
torchrun --standalone --nproc_per_node=4 \
  op_tests/multigpu_tests/bench_mega_moe.py \
  -tpr 16384 --layers 8 -e 384 -k 6 -hd 5120 -id 2304 \
  --fused_shared_experts 0 --routing uniform \
  -q a4w4_mxfp4 --dispatch_wire fp4 \
  --stage1_fused true --stage2_fused true \
  --profile_table 1 --prof_replays 32 --acc_verify 0 \
  --warmup 5 --iters 32 --seed 0 --data-init uniform
```

For the push control, change only `AITER_MEGA_DISPATCH_PULL=0`. Keep S1F off,
routing, CSV, timing method, and all other workload settings identical.
`-tpr` means tokens **per rank**: this case has 65536 global tokens. The EP
CSV lookup uses 96 local experts, `topk=-1`, and token key 65536 for this case.
Do not equate an EP token key with the non-EP local-token workload.

## Validation

- CPU recipe tests: `python3 -m pytest -q tests/test_mega_moe_gfx1250_recipe.py`
  passed, 33 tests, on 2026-11-17.
- The existing multi-GPU test now includes `_check_pull`. It compares full
  output with push over four changing-input graph replays, alternating random
  routes and routes concentrated on rank 0. It also checks exact valid-row
  payload, scales, routing weight bits, and the shared local preshuffle path.
- The current non-persistent TDM implementation passed that check at EP4,
  16384 tokens/rank, H5120/I2304/E384/top-k 6, with preshuffle enabled.
  Saved log on the development host:
  `/workspace/benchmarks/ds41_pull_grid_20261116/check_target_shuffle1/test.log`.
- The standard suite is
  `python3 op_tests/multigpu_tests/test_mega_moe_gfx1250.py --world 4`.
  Set `AITER_MEGA_PULL_TDM_GATHER=1`, `AITER_MEGA_QUANT_FUSE=128`, and
  `AITER_MEGA_S1F=0`, but leave `AITER_MEGA_DISPATCH_PULL=0`: the pull check
  enables it internally, while the suite also exercises FP8 push recipes.
  Its default shape differs from the saved DSV4.1 Flash validation shape. GPU jobs should run
  only after an idle check; use the shared `gpu-lock` wrapper when installed.

## Performance evidence and limits

The following are **historical persistent-kernel measurements**, not timings
of the current non-persistent TDM kernel. Workload and routing match the
prefill command above. Units are microseconds; E2E is latency per layer.

| Version | Runs | Plan / plan+quant | Dispatch | GEMM1 | GEMM2 | E2E/layer |
|---|---:|---:|---:|---:|---:|---:|
| Push | 1 | 63.39 | 252.04 | 620.62 | 585.34 | 1931.84 |
| Original 512-thread persistent pull, shuffle | 2 | 126.07 | 340.43 | 504.27 | 578.35 | 1942.69 |
| 128-thread persistent pull, two gathers, shuffle | 2 | 126.45 | 306.39 | 503.59 | 578.08 | 1904.61 |
| Original 512-thread persistent pull, no shuffle | 1 | 125.87 | 335.57 | 631.30 | 584.15 | 2068.29 |
| 128-thread persistent pull, two gathers, no shuffle | 1 | 125.38 | 290.10 | 633.79 | 584.62 | 2024.81 |
| Current 128-thread non-persistent TDM pull | 0 | pending | pending | pending | pending | pending |

Kernel timings used HIP event nodes inside a graph: median over 256 samples
per GPU, averaged over four GPUs, then averaged over successful runs. E2E
used the separate uninstrumented native graph. Push's plan column excludes
its separate quantization; pull's plan column includes quantization and
inverse-map writes. Columns therefore do not sum to E2E.

The earlier two-gather persistent change reduced shuffled dispatch by 10.0%
and observed E2E by 2.0% versus original pull. Its dispatch was still 21.6%
slower than push. The small E2E difference versus push is not sufficient to
claim a stable win from one push run. Full-pipeline timing for the current
non-persistent kernel remains unavailable; the isolated comparison below
shows a large regression.

Raw historical records and both historical sources are preserved on the
development host in `/workspace/benchmarks/ds41_pull_inflight_20261115/`.
These local artifact paths are not bundled with the repository.

A follow-up comparison in
`/workspace/benchmarks/ds41_pull_dispatch_only_20261117/` uses identical
source/map/output buffers for the 128-thread persistent and non-persistent
kernels. It prepares real plan/quant inputs once and times eight dispatches
per graph replay with external HIP events, over 32 interleaved trials.
It checks exact valid-row bytes and requires eight real captured kernel
nodes before accepting timing. This is isolated dispatch with producers
already complete, not E2E or in-pipeline timing. The first attempt captured
empty graphs; all those timings are invalid. Full-pipeline stalled attempts
are also excluded.

The corrected comparison completed on 2026-11-17 and was independently
repeated in `/workspace/benchmarks/ds41_pull_dispatch_only_20261117_confirm/`
against commit `51e9923e7`. All four GPUs passed the PyTorch health check and
exact valid-row payload/scale comparison. Each captured graph contained the
expected eight kernels with the correct grid and 128-thread workgroups.

| Isolated dispatch variant | Grid TGs | Run 1 (us) | Run 2 (us) | Mean (us) |
|---|---:|---:|---:|---:|
| 128-thread persistent pull | 256 | 275.01 | 279.09 | 277.05 |
| 128-thread non-persistent TDM pull | 6528 | 3315.61 | 3260.46 | 3288.04 |

Each run reports the mean of four per-GPU medians over 32 interleaved trials.
Removing persistence increases isolated dispatch latency by **11.87x**
(1086.8% regression). This result does not include plan/quant or GEMMs and
must not be compared directly with the historical in-pipeline push timings.
That version repeated startup readiness and metadata work for every TG,
including unused TGs; LDS remained 180608 bytes/TG.

## One-time readiness synchronization

The current workspace moves TDM readiness into `ep_dispatch_pull_ready_ep4`
(one 32-thread TG for EP4), followed by the unchanged 6528-TG copy grid on
the same stream. This preserves publication/acquisition before remote reads,
including when a rank has no live destination rows. The combined launcher
executes both kernels on every dispatch. The legacy async path is unchanged.

The isolated benchmark includes **both sync and copy**, measured over eight
complete dispatches per graph replay and 32 interleaved trials. Graph checks
require eight ready nodes and eight copy nodes. Producers are already
complete, so these timings do not include actual plan/quant latency or E2E.

| Implementation | Dispatch total (us) |
|---|---:|
| Persistent baseline, same run | 274.10 |
| Non-persistent TDM, one-time sync + copy | 182.26 |

The synchronized non-persistent path reduces latency by 33.5% against the
same-run persistent baseline. Results and source snapshot are in
`/workspace/benchmarks/ds41_pull_sync_once_20261117/` on the development host.

A four-GPU full-MoE correctness run completed on 2026-11-17, 17:30:39 to
17:31:03 CST, at 16384 tokens/rank, H5120/I2304/E384/top-k 6. Four changing-input
graph replays passed, including random and rank-0-concentrated routing,
full output against push, exact payload/scales/routing weights, and the local
preshuffle reference. Exit code was zero, the host boot ID was unchanged,
and `rocm-smi --showpids` reported no GPU processes afterward. This single run
did not reproduce the reported machine hang; it does not establish the cause
of previous hangs. Log: `health_recheck/test.log` in that artifact directory.

## Opt-in S1F pull with A preshuffle

Set `AITER_MEGA_S1F=1`, `AITER_MEGA_DISPATCH_PULL=1`,
`AITER_MEGA_PULL_TDM_GATHER=1`, and `AITER_MEGA_S1F_PULL=1` to select the
pull/shuffle producer in the original `make_s1f_stage1` kernel. It requires
fused-scatter combine, no bias, and M64/128/256. The producer count defaults
to64 of256 resident workers. Ordinary push uses the unchanged mailbox
producer. Pull forwards the resolved CSV tile sizes, wave grid, buffer count,
next-stage prefetch, TDM wave count/cache hint, LDS interleave, and drain setting
to the shared GEMM consumer. S1F still requires four waves and cluster1x1;
requested clusters are logged and fall back to1x1. CSV persistent_workers=0
retains256 resident S1F workers on this256-CU device. Task claims, A-ready
gates, entry generations and parity-split queue heads remain shared with push.
Planar LDS or next-stage prefetch uses ordinary C-store retirement; the default
nonplanar/no-prefetch path retains deferred retirement.

The path is compact plan+quant/inverse-map publication, cross-rank
`s1f_pull_ready_ep4`, fused pull+shuffle+GEMM1/activation/output quant,
then GEMM2+scatter and combine. Source readiness remains outside the fused
kernel: a peer may read the quantized input only after its source plan
finishes. The ready kernel also converts psum into the original64-row
`tile_info` schema. Each tile names its local pull publisher in the source
mask. Dedicated `pull_tile_flag` storage keeps the pull generation independent
of ordinary push launches.

Each producer wave owns a complete64-row flag tile, gathered in four M16
rounds with the shared `make_pull_tile` helper. Each round issues two
full-wire gather8 operations before waiting, shuffling and writing back.
The four waves have private M16 LDS slices (64rows in total per round).
After its stores retire, the owning wave publishes the original relaxed
system-scope tile flag; no per-tile WG barrier or system fence is added.
The original two producer-to-consumer transition barriers remain in place.
For M256 the consumer waits on four64-row flags exactly as push does.

GEMM and copy alias one arena, sized to the larger requirement, followed
by the original256B control block. One CTA per CU permits up to320KiB.
The shared row-end pitches are2832B for H5120 and3856B for H7168, both
`64*n+4` DWORDs. Four M16 slices use181248B and246784B respectively;
these sizes are not added to the GEMM arena. Each source row retains
payload, scales and wire alignment padding; TDM reserves only an additional
16B at the LDS row end. A uses M16/K32 packed payload and row-major scales.

After merging main9eb34ee0b (merge209e47f18), two interleaved push/pull
pairs at EP4/tpr16384,H5120/I2304,E384/top6,balanced,8layers,5warmup,
32iterations measured the following4-GPU averages (microseconds):

| Scope | Push S1F | Pull + shuffle S1F |
|---|---:|---:|
| Plan + quant |36.18|104.15|
| Source-ready + tile metadata |—|13.19|
| Fused GEMM1 including producer |626.36|612.62|
| GEMM2 + scatter |593.11|591.23|
| E2E/layer |1645.94|1709.15|

The fused stage improves2.19%, but total latency regresses3.84% because
plan/ready costs outweigh that gain. GEMM2 is effectively unchanged.
E2E uses native replay medians; kernel intervals use explicit HIP events.
Merged-main correctness passed H5120/I2304/tpr16384 with changing graph
inputs, empty receivers, exact payload/scales/weight checks and output
comparison to push. Full commands, symbols, run variability and review
records: `/workspace/gpubench/s1f_shared_consumer_20261118/README.md`.

The compact-plan optimization subsequently stages pull inverse metadata by
segment in LDS and writes consecutive8-byte records to each peer. It overlaps
local staging with histogram exchange and omits the unused final tok_map output;
destination-base synchronization and source visibility remain unchanged.
The measured planner uses118336B LDS (43520+9280+8192*8).

At the same EP4/tpr16384 balanced shape, two optimized pull runs measured
plan+quant39.43/39.61us (mean39.52), versus a fresh push36.10us. Pull E2E was
1651.04/1650.34us (mean1650.69), versus fresh push1648.78us. Relative to the
previous pull above, plan time decreases62.05% and E2E decreases3.42%.
Plan remains3.42us slower than push; total E2E is effectively comparable
(+0.12%), not a demonstrated pull win. Ready+metadata12.44us, fused GEMM1
615.54us and GEMM2+scatter592.95us are the optimized pull averages.
Single-block H7168/tpr17 and multi-block H5120/tpr16384 EP4 checks passed,
including changing graph inputs and exact metadata/payload validation.
Commands, per-run results and review record:
`/workspace/gpubench/compact_pull_opt_20261118/README.md`.

The subsequent three-slot pipeline/M256-publication experiment was reverted
at the user's request. It passed EP4 correctness but regressed at the same
balanced tpr16384 shape. Two runs averaged
fused GEMM1 740.94us and E2E 1778.54us, versus615.54us
and1650.69us before the producer change. In that experiment each wave published an entire M256,
which delays initial readiness and distributes384groups unevenly across256
producer waves (256 or512rows/wave instead of384). These scheduling differences
have not been independently timed. Full table, review and source snapshots:
`/workspace/gpubench/s1f_pull_pipeline_20261118/README.md`.

Historical measurements below predate the shared-consumer substitution:

The shared S1F helper passed the existing EP4 checks for H5120/I2304 at
16384 tokens/rank and H7168/I1024 at17 tokens/rank, including four changing
graph inputs and exact source payload/scales/weights checks. At H5120,
balanced EP4/tpr16384, the old S1F pull slab version versus the shared
row-end version measured632.50->630.25us for fused GEMM1 and
1730.68->1731.25us E2E/layer: performance is effectively flat.
The old slab pitch also satisfied64*n+4 DWORDs. Full records are in
`/workspace/gpubench/pull_lds_padding_20261118/s1f_rowtail_results.md`.

Source snapshots, per-run commands, native E2E and explicit HIP graph event
results are in `/workspace/gpubench/s1f_pull_20261117/`. Its README separates
the earlier K1024 experiment from the current LDS-derived K1152 version.
Fused stage1 timing includes pull+shuffle+GEMM1; a separate non-S1F GEMM1
interval excludes dispatch, so those columns have different scopes.

### Pull shuffle LDS padding

The payload transpose uses an LDS row pitch of `64*n+4` DWORDs to spread
16 adjacent rows across the64 banks. This does not change the source wire
or the GEMM-facing payload/scale layout.

Standalone Flash pull retains its two full-wire gather8 operations per wave.
The source wire remains unchanged. TDM periodic padding is disabled; the
descriptor uses the LDS row pitch as tile dim0 and the wire width as tensor
dim0. The trailing OOB columns reserve the row-end gap. Payload and scales
retain their source offsets within each LDS row, with no internal gaps.
For wire2816B the LDS pitch is2832B (`64*11+4` DWORDs).

Standalone pull uses exactly64 rows and128 threads per workgroup: four
waves, each owning16 rows and issuing two gather8 operations before waiting.
The grid is `ceil(row_capacity / 64)`. K5120 uses181632B LDS, including96
expert ends. For K7168, wire3840B has a padded pitch of3856B
(`64*15+4` DWORDs), and the64-row tile plus metadata uses247168B.
Both fit the gfx1250 hardware320KiB budget. There is no K slab fallback;
a wire whose full64-row tile cannot fit is rejected.
Source readiness and the output payload/scale layout are unchanged.
The fixed64-row path passed the existing EP4 K5120/I2304 check at17
tokens/rank. The earlier K7168/I1024 row-end check passed the same64-row
kernel geometry and body. Both checks include four changing-input graph
replays and exact source payload/scales/weights checks. Bank-address enumeration
for both widths shows
unique banks within each16-lane payload-vector group and each32-lane scalar
scale load; hardware bank-conflict counters were not measured.
S1F reuses the same full-wire row-end helper. Cluster code remains reverted.
Final H5120 standalone timing at EP4/tpr16384, using32 interleaved HIP-event
trials, measured ready-sync+pull+shuffle at203.44->201.09us for balanced
routing and180.65->178.17us for random uniform routing (mean GPU medians).
The baseline is unpadded pull; the new version adds only row-end LDS padding.
Exact payload/scales matched. Plan/quant and MoE E2E are outside this timing.
The older standalone harness omitted the balanced argument; its historical
results used random uniform routing despite the environment setting.
Results and source snapshots, including the earlier implementations, are in
`/workspace/gpubench/pull_lds_padding_20261118/`.


## CSV-fed pull S1F, 2026-11-18

At EP4, `-tpr16384` means65536 global tokens. The actual lookup selects
`aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv`, token65536,
expert96, topk=-1, merged into `/tmp/aiter_configs/tuned_grouped_fmoe.csv`.
The32768 warmup lookup is not this step's selected row. GEMM2's wpt1 is correct.

GEMM1 now uses256x256x256,2x2 waves,nb4,prefetch1,interleave1,wpt2;
cluster4x2 falls back to1x1 as required by the explicit-tile S1F scheduler.
GEMM2 remains256x256x256,2x2 waves,nb4,prefetch1,interleave1,wpt1,cluster1x4.
Both kernels reserve327680B LDS. The S1F control block occupies unused planar
B-alignment padding [181248,181504), beyond producer storage and GEMM C stores.
Empty M64 metadata entries carry neither live rows nor a source-ready mask.

Balanced EP4/H5120/I2304/E384/top6,8layers,5warmup,32iters; microseconds,
averaged across ranks. Kernel values are explicit HIP-event intervals, including
boundary overhead. Native E2E is measured before inserting event nodes.

| Version | Plan+quant | Ready+metadata | S1F GEMM1 incl producer | GEMM2+scatter | E2E/layer |
|---|---:|---:|---:|---:|---:|
| Prior push control (existing measurement) |36.10|0|625.54|597.04|1648.78|
| Prior pull before CSV feed (2 existing runs) |39.52|12.44|615.54|592.95|1650.69|
| Pull with CSV (2 new runs) |39.96|12.65|570.04|595.48|1609.77|

GEMM1 latency fell7.39%, E2E2.48% versus prior pull. The push control is the
same merged workspace with pull disabled, not a separate clean-main checkout.
Correctness passed at tpr16384/H5120/I2304 and tpr17/H7168/I1024, including
four graph input changes, hot/empty ranks, exact dispatched payload/scales and
routing weight checks. Eight-wave CSV configurations remain unsupported by S1F.

Producer scheduling did not change:64 producer TGs with4 waves each; each
wave owns an M64 flag tile and copies it as four M16 chunks. Each chunk issues
two gather8 operations then waits tensorcnt0 before shuffle/store. Thus each
wave has gather-empty intervals during writeback; CSV nb4 only changes GEMM
buffering. Different waves can overlap, so this does not establish a measured
TG-wide idle fraction. The previously rejected producer pipeline remains reverted.

Reproduction, symbols, source snapshots, correctness logs and both timing runs:
`/workspace/gpubench/s1f_pull_csv_20261118/`.


## Split-barrier S1F pull producer, 2026-11-18

This experiment used a TG-uniform M256 loop. Each wave finishes its
M64, retires stores/LDS reads, signals the TG barrier, and gathers the next
round's first M16 into its private LDS slice BEFORE waiting. After the wait,
each wave publishes its existing M64 flag and consumes the prefetched chunk.
The other three M16 chunks retain the ordinary gather/wait/shuffle/store order.
Empty/tail waves participate in every barrier round. Final rounds do not
prefetch. Consumer flags, GEMM CSV settings and LDS allocation are unchanged.

This experiment temporarily replaced the independent-wave boundary scheduling
described in the CSV-only result above. It reuses the common gather/writeback operations via
`make_pull_tile_ops`; the unfused wrapper retains its previous sequence.

EP4/tpr16384, balanced H5120/I2304, same native/HIP-event harness:

| Run | Plan+quant | Ready | GEMM1+producer | GEMM2+scatter | E2E/layer |
|---|---:|---:|---:|---:|---:|
| CSV-fed baseline,2-run mean |39.96|12.65|570.04|595.48|1609.77|
| Split barrier,run1 |44.96|13.17|570.05|601.81|1623.97|
| Split barrier,run2 |39.86|12.58|570.02|596.33|1612.02|

No measurable fused-GEMM1 benefit. E2E did not improve; other unchanged stages
also varied, so these timings do not isolate the barrier's cost. Correctness
passed both fused/unfused H7168/I1024/tpr17 and fused H5120/I2304/tpr16384,
including changed graph inputs and hot/empty ranks. No hang/reboot observed.
The implementation was reverted to the best measured CSV-fed baseline before
commit; baseline snapshots, patch, review log and all timings are in `/workspace/gpubench/s1f_pull_split_barrier_20261118/`.
