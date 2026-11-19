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
S1F pull with A preshuffle, uses 96 producers after the ratio tuning below, and enables fused GEMM2. The
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
| **Full-row pull, 64 producers, two-run mean** | **39.96** | **12.65** | **570.04** | **595.48** | **1609.77** |
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
to the shared GEMM consumer. S1F still requires four waves. Cluster mode is opt-in; when disabled,
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
cluster4x2 falls back to1x1 in these measurements, with consumer clustering disabled.
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

## Producer-count tuning, 2026-11-18

Commit 13b230a52, full-row gather, EP4, tpr16384/global65536, E384, top6, H5120/I2304, balanced, 8 layers, 5 warmups, 32 iterations. Both GEMMs use A preshuffle. Only producer count changes; GEMM1 cluster remains 1x1.

Native E2E median per rank divided by 8, then averaged over ranks. Stage values are mean HIP-event intervals including boundary overhead; stages do not sum to native E2E.

| Producers | Initial consumers | Runs | Plan+quant us | Ready us | GEMM1 incl producer us | GEMM2+scatter us | E2E/layer us |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 224 | 1 | 39.48 | 12.46 | 610.94 | 594.75 | 1650.94 |
| 64 | 192 | 3 | 40.16 | 12.52 | 570.01 | 595.52 | 1609.81 |
| 80 | 176 | 1 | 39.82 | 12.49 | 564.18 | 594.80 | 1604.87 |
| 96 | 160 | 3 | 39.72 | 12.51 | 561.38 | 595.95 | 1601.36 |
| 112 | 144 | 1 | 40.09 | 12.60 | 563.73 | 595.63 | 1604.35 |
| 128 | 128 | 1 | 39.67 | 12.60 | 567.24 | 595.18 | 1604.58 |
| 192 | 64 | 1 | 39.42 | 12.53 | 605.32 | 595.80 | 1645.44 |

96 producers reduce GEMM1 latency by 1.51% and E2E by 0.52% versus 64, based on three runs each. Other candidates have one screening run. This does not establish a global optimum over all counts/shapes.

Baseline E2E range: 1608.55–1612.07 us; 96-producer range: 1599.57–1603.07 us. Producers join the consumer queue after their own work; these ratios describe initial roles.

Full EP4 correctness passed with 96 producers: changing graph inputs, random/hot/empty ranks, full output vs push and exact dispatched payload/scales/weights. All runs exited 0 with unchanged boot ID. Every run directory contains three prelaunch review passes, source snapshots, command, environment and idle checks.

```bash
BENCH_PRODUCERS=96 bash /workspace/gpubench/s1f_pull_ratio_20261118/run.sh current p96_new
```

## Optional TG0-led 1xN clusters for initial S1F consumers

Enable with `DS41_CLUSTER=1 bash run_ds41_flash_best.sh`, or set
`AITER_MEGA_S1F_PULL_CLUSTER=1`. Default is 0. GEMM1 keeps the CSV cluster_n and
projects cluster_m to 1; Flash's requested 4x2 becomes 1x2. Incomplete N groups
or physical role groups fall back to 1x1. GEMM2's cluster configuration is unchanged.

## Execution contract

- Only initial consumers cooperate as physical 1xN clusters. CSV cluster_m is
  projected to 1; cluster_n is retained if N tiles and worker/consumer counts
  form complete clusters. Unsupported geometry falls back to 1x1. The shared
  GEMM builder still supports ordinary non-S1F MxN clusters.
- TG0 atomically acquires one `(m_tile, n_base)` group from the shared dynamic
  queue. Followers receive the same task and use `n_base + local_n`; all use
  the same shuffled A tile and expert. No fixed producer/consumer work partition.
- TG0 alone checks the original producer flags for A and its scales, including
  the normal next-task flag prefetch. It release-publishes one A-ready epoch.
  Followers acquire that epoch before any A/A-scale TDM issue. This is required
  even though A uses multicast: early-timeout can serve requests before all
  peers have arrived, so multicast alone is not a producer-ready barrier.
- `a_ready` and `item_hooks` remain active for both roles. TG0's next group claim
  and metadata/flag prefetch overlap the existing drain/epilogue stages.
- Producer dispatch and its original store/flag retirement are unchanged. After
  dispatch, a producer TG takes groups from the same queue, computes all N
  columns independently with multicast disabled, then claims another group.
- The GEMM body retains its ordinary TDM multicast and LDS/tensor waits. There
  is no added software cluster barrier. Task and A-ready publication are the
  explicit leader/follower protocol, not duplicated producer-flag polling.

## Mailbox lifetime and storage

Each initial consumer cluster has one write-once record per possible task plus
an exhaustion sentinel. A record contains task code, task epoch, A-ready epoch
and one padding dword. Records are not recycled inside a kernel invocation:
TG0 cannot overwrite data that a delayed follower has not consumed. The launch
epoch rejects prior graph replay contents. All configurations retain the same
S1F control/generation counters; private mailbox tensors are cached per resolved
configuration. In the pull adapter, the otherwise unused `addr_runs` parameter
carries this mailbox. Push retains its original meaning for that argument.

The conservative allocation is
`16 * initial_consumer_clusters * (ceil(compact_cap/tile_m) * N_tiles/N + 1)`
bytes per cached clustered configuration. It is global memory, not additional
LDS. This prioritizes a simple bounded protocol over minimizing scratch space.

## Validation and one recorded failure

43 CPU recipe tests pass, including 1xN projection, fallback and mailbox bounds.
An independent CPU audit checks task coverage and delayed-reader record lifetime.
EP4 GPU checks pass for actual 1x2/1x4, MxN-to-1xN projection, Flash
H5120/I2304/tpr16384, and H7168/I1024/tpr17 with 1x4. Tests compare four changing
input graph replays, hot/empty ranks, output against push, and exact dispatched
payload/scales/routing weights.

The first full Flash correctness attempt (`current_check_flash_r1`) reported
GPU memory-access faults and then failed allgather. Its log did not reach the
new pull-S1F GEMM configuration message. Root cause remains unconfirmed.
The unchanged-source retry with phase tracing passed; a further original check
without the diagnostic synchronization also passed. The performance run and
K7168 check passed afterward. No GPU reset was performed; boot IDs are unchanged.
The failed run is retained and is not counted as a successful validation.

Every GPU launch had three recorded reviews and a successful immediate idle
check, polling each second while busy. Production source was frozen during runs.
After the timing run, only formatting/comments and a host guard rejecting
clustered calls without the pull adapter changed; the final K7168 check used
that source.

## Flash timings

DSV4.1 Flash, EP4, tpr16384/global65536, E384, top6, H5120/I2304, balanced,
96 producers/160 initial consumers, S1F pull, fused GEMM2, both GEMMs A preshuffled,
8 layers, 5 warmups, 32 iterations. Microseconds, four-rank averages.
E2E is native median replay time / 8; GEMM stages use explicit HIP graph events
with boundary overhead and must not be summed to E2E. One run per row.
The first two rows reuse the earlier native-sync measurements; only the TG0
leader row is newly measured here.

| Implementation | GEMM1+producer us | GEMM2+scatter us | E2E/layer us |
|---|---:|---:|---:|
| Earlier 1x1 control | 560.19 | 597.30 | 1602.47 |
| Earlier fixed-partition 1x2 | 697.28 | 597.08 | 1738.47 |
| New TG0 leader 1x2 | 605.94 | 595.06 | 1641.85 |

New plan+quant: 39.55 us; source-ready kernel: 12.30 us.

Compared with the fixed partition, GEMM1 decreases 13.10% and E2E decreases
5.56%. It remains 2.46% above the earlier unclustered E2E measurement, so the
feature stays opt-in/default off. Task/ready publication and group-size scheduling
still differ from unclustered S1F; their individual timing costs are not isolated.


Logs, source snapshots, commands and event samples:
`/workspace/gpubench/s1f_cluster_leader_20261118/`.

## Optional mixed-role S1F affinity, 2026-11-18

Set `AITER_MOE_S1F_AFFINITY=1` alongside the normal S1F pull options.
The reproducer preserves this switch:
`AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh`.
The default remains off. This option takes precedence over the earlier
consumer-only `AITER_MEGA_S1F_PULL_CLUSTER` experiment.

The total producer/worker ratio is preserved. With 96 producers and 256
workers, its reduced ratio is 3 producers + 5 initial consumers per physical
8-TG cluster: 32 clusters, hardware dimensions `(4, 2, 1)`. Ratios reducible
to 2/4/8/16-TG clusters are supported; unsupported ratios raise before launch.
GEMM tiling and its four waves are unchanged. This variant uses physical
clustering for locality, without GEMM multicast or added cluster barriers.

M tile `m` belongs initially to queue `m % cluster_count`. Producers in that
cluster gather/shuffle its A and scales, and publish the existing M64 flags.
Its consumers claim N tiles from that queue first. After exhaustion, any TG
may steal from other queues. Producers join GEMM after finishing dispatch.
A task is claimed once through its owner's atomic queue even when stolen;
no duplicate task bitmap or leader/follower task mailbox is needed. A TG
that has consumed A may reuse its ready state while advancing N for the same
M tile. Each launch's existing source-ready kernel resets the queue heads;
32 heads separated by 128 B add 4096 B of global scratch, no additional LDS.

At Flash EP4, tpr16384/global65536, balanced routing, each GPU has 384 valid
M256 tiles and 6912 GEMM1 (M,N) tasks. The 32 queues initially own 12 M tiles
each. This ownership is a preference; cross-cluster stealing remains enabled.
Sharing a cluster/L2 does not guarantee cache residency or a 100% hit rate.
The change also changes task order and ready reuse, so these results do not
isolate cache locality as the sole cause.

Fresh interleaved runs, 8 layers, warmup5/iters32, means across 4 GPUs and
3 runs. Native E2E is measured before HIP event insertion. GEMM1 includes
its producer; GEMM2 includes scatter. Event intervals include boundary
overhead and must not be summed into native E2E. Actual runtime CSV lookup
is token65536, topk=-1; token32768 also appears during precompile setup.

| Mode | GEMM1+producer us | GEMM2+scatter us | E2E/layer us |
|---|---:|---:|---:|
| Affinity off | 555.74 | 582.78 | 1583.32 |
| Affinity on, 3P+5C | 534.26 | 582.38 | 1567.34 |
| Latency reduction | 3.86% | 0.07% | 1.01% |

GEMM1 improves in all three runs. E2E ranges overlap: off1576.47–1588.34 us,
on1556.35–1584.15 us, so the smaller E2E improvement needs a variability
qualification; default selection has not changed.

Validation: 53 CPU recipe tests; EP4 changing-input output checks plus exact
payload/scales/routing-weight comparison against push, at H5120/I1024/tpr17
and full Flash H5120/I2304/tpr16384. Four random/hot routing graph replays
include empty ranks/experts. Both checks and six performance runs exited0,
with unchanged boot ID. Two earlier host configuration failures are retained:
the small Flash CSV selected unsupported prologue A-scale loads; a test-only
persistent-worker override conflicted with the ordinary push reference's
cluster_m=2 row and was removed. Neither is counted as a passing test.

Commands, source snapshots, all samples, symbols and reviews:
`/workspace/gpubench/s1f_mixed_affinity_20261118/`.

## Completed BTH tuning with affinity, 2026-11-18

Keep BTH=0. No repeatable E2E benefit from another hint was established.
All rows below are fresh measurements with affinity enabled, P96/G256,
tpr16384/global65536, balanced EP4, both GEMMs using preshuffled A.
The override affects only B TDM loads, in both GEMM1 and GEMM2.

| BTH | Hint | Runs | GEMM1 + producer (us) | GEMM2 + scatter (us) | E2E/layer (us) | E2E reduction vs 0 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | regular | 4 | 536.86 | 590.54 | 1592.91 | +0.00% |
| 1 | non_temporal | 1 | 552.47 | 612.42 | 1624.28 | -1.97% |
| 2 | high_temporal | 3 | 537.55 | 590.17 | 1594.89 | -0.12% |
| 3 | last_use | 1 | 550.20 | 609.37 | 1618.16 | -1.59% |
| 4 | non_temporal_regular | 1 | 549.64 | 608.72 | 1605.96 | -0.82% |
| 5 | regular_non_temporal | 3 | 539.55 | 591.45 | 1590.52 | +0.15% |
| 6 | non_temporal_high_temporal | 1 | 552.16 | 613.14 | 1611.54 | -1.17% |

The comparison uses four fresh BTH=0 repeats. Its first run was unusually
slow (GEMM1 546.82, GEMM2 758.16, E2E 1789.61 us) with large rank differences.
That run remains in raw results and the all-run summary, but is excluded
from this comparison; the cause is not established. No gain is claimed
against that anomalous baseline. Values 1/3/4/6 had one screening run each;
0/2/5 were repeated and interleaved because their differences were small.

BTH=5's mean E2E reduction is only about 0.15%, with slower mean GEMM1 and
GEMM2. The last adjacent 0/5 pair reversed it: 1588.12 vs 1596.22 us.
BTH=2 also failed to sustain its initial advantage. This does not justify
changing the default B-load policy. Hints affect B-weight cache treatment;
they do not change producer/A-data affinity, dispatch overlap, or readiness.
There are no cache-counter measurements attributing the differences.

All 15 performance runs exited successfully and retained the same boot ID.
Both actual GEMM symbols reflect every nonzero BTH override, and launch
geometry/LDS match across the sweep. These runs used acc_verify=0; no new
numerical validation was performed in this hint-only sweep. The unchanged
BTH=0 affinity kernel retains the prior passed full Flash reference checks.
No GPU kernel source or CSV was changed during this tuning.

Reproduce native E2E with the retained setting:

```sh
AITER_MOE_S1F_AFFINITY=1 DS41_BTH=0 bash /app/aiter/run_ds41_flash_best.sh
```

Set DS41_BTH=2 or 5 to reproduce the alternatives. The script clears raw
AITER overrides, so use DS41_BTH with this script. Raw benchmark launches
can instead use AITER_TDM_B_TH directly. Logs, run-level/rank-level values,
exact symbols, commands and diffs are under
/workspace/gpubench/s1f_affinity_bth_20261118/.

## Optional instruction-cache prefetch (zxe port)

`AITER_FLYDSL_INST_PREFETCH=1` enables the shared implementation ported from
`zxe/mega_update_moe` commit `a85897e1b`. Each hardware wave enables
MODE.SCALAR_PREFETCH_EN and prefetches its share of 4 KiB instruction chunks,
capped at 56 KiB and bounded by the assembler-resolved function end.

It runs once at entry for push/pull S1F, static grouped GEMM, and standalone
persistent grouped GEMM. Persistent task bodies skip it. Enabled GEMM symbols
contain `_ipf`. Data TDM scheduling, LDS layout and ready flags are unchanged.

Leave the variable unset to retain the original single-wave S1F tail prefetch
and original standalone GEMM behavior. Explicit `0` disables software
instruction prefetch, including the old S1F helper; `1` selects the new helper.
Use separate processes because this switch is read at import time.

```sh
# Original policy (default).
env -u AITER_FLYDSL_INST_PREFETCH AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
# New prefetch experiment.
AITER_FLYDSL_INST_PREFETCH=1 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
```

Measured 2026-11-19: EP4, 16384 tokens/rank (65536 global), balanced Flash,
P96/C160, affinity8p3, unchanged CSV, eight layers, warmup5, iters32. Three
interleaved runs per variant; values are microseconds averaged over four GPUs
and three runs. E2E is the native graph-replay median per layer; other columns
are explicit HIP-event intervals in a separate pass and do not sum to E2E.
GEMM1 includes its producer and GEMM2 includes scatter.

| Policy | E2E/layer | GEMM1 | GEMM2 | Compact plan | Ready |
|---|---:|---:|---:|---:|---:|
| Original commit c3686f50d | 1582.15 | 539.89 | 584.44 | 40.67 | 13.24 |
| Software prefetch off | 1590.48 | 557.80 | 583.84 | 39.81 | 13.24 |
| New distributed prefetch | 1585.45 | 542.19 | 593.40 | 39.85 | 12.99 |

New vs original: E2E +0.21%, GEMM1 +0.43%, GEMM2 +1.53% latency. E2E run
ranges overlap: original1562.67–1594.84, new1568.44–1599.04. No overall gain
was established, so the old policy remains the default. Static GEMM2 adds the
new prologue to32640 workgroups; its extra instruction/register cost is a
plausible explanation for the slowdown, not a counter-verified diagnosis.

Validation: four-GPU PyTorch health passed; small and full Flash changing-input
output/payload/scale checks passed. Full Flash checks also passed for both the
final explicit-on and default policies. Nine timing jobs exited successfully.
Helper AST matches zxe; isolated helper compilation, syntax, Ruff F, shell and
diff checks passed. The failed consumer-pair experiment was removed from the
focused diff and preserved in `before.patch` under the artifact directory.

Detailed report, per-run/per-rank timings, symbols, CSV/source snapshots,
reproduction runners and preserved experimental patch:
`/workspace/gpubench/s1f_instruction_prefetch_20261118/REPORT.md`.

## Experimental producer-leader readiness

`AITER_MOE_S1F_READY_FIRST=1` removes the standalone ready/setup kernel from
S1F pull. Default remains `0`: no E2E speedup has been established. This switch
changes neither push nor non-S1F dispatch. Use separate processes for comparisons.

The current protocol is local-first with one producer leader per GPU; it does
not use rank-mask scheduling:

1. The existing compact-plan + quant kernel also initializes pull tile metadata,
   local M16 intervals and affinity queue heads. It invalidates the local
   all-ready epoch and records the expected epoch. It does not publish source
   readiness while quantization may still be in flight.
2. At S1F entry, producer slot 0 publishes source-ready after the complete local
   plan + quant kernel. It immediately system-acquires the other source epochs.
   A workgroup barrier transfers these observations to its publishing thread,
   which agent-release-stores the epoch into one local GMEM all-ready flag.
3. The leader then processes all its ordinary producer-owned tiles. Other
   producers can first copy their local-only M16 chunks. Before remote copies,
   thread 0 of each TG agent-acquires the shared local all-ready flag, followed
   by a TG barrier. They never independently poll peer-ready flags. No readiness
   check is added to every token, and ready is latched for the remote drain.
4. Each M64 still has exactly one wave owner. That wave retires data stores
   before publishing the original consumer tile-ready flag. Consumer scheduling,
   GEMM layout, and producer-to-consumer transition are unchanged.

There is one all-ready flag per destination GPU, not one shared across GPUs.
Source-ready and consumer tile-ready flags have separate roles and are retained.
The three local words are source epoch (+0), all-ready epoch (+4), and expected
plan epoch (+8). The plan resets all-ready to epoch-1 before each invocation.
Metadata is generated from the existing histogram; expert/source-major ordering
is unchanged. A mixed-source M16 waits for all-ready; a partial M16 can run early
only when all its valid rows are local. No extra routing padding is introduced.
Worker count is limited to the CU count and 256-head capacity for this path.

```sh
AITER_MOE_S1F_AFFINITY=1 AITER_MOE_S1F_READY_FIRST=1 bash run_ds41_flash_best.sh
# Original separate-ready control:
AITER_MOE_S1F_AFFINITY=1 AITER_MOE_S1F_READY_FIRST=0 bash run_ds41_flash_best.sh
```

Before the histogram-first follow-up below, measured 2026-11-19: EP4,
16384 tokens/rank (65536 global), E384/top6/H5120/I2304,
unchanged model CSV, original instruction-prefetch, eight layers, warmup5/iters32,
uniform input and seed0. P96/C160 uses affinity8p3; P64/C192 uses affinity4p1.
Values are microseconds per layer, averaged over four ranks. Each row is one run;
controls are refreshed within this experiment. Native E2E and HIP graph-event
stage timing are separate passes and must not be added together.

| Routing / version | E2E/layer | Plan + quant | Standalone ready | GEMM1 + producer | GEMM2 + scatter |
|---|---:|---:|---:|---:|---:|
| Balanced / separate ready / P96 | 1588.67 | 39.27 | 12.74 | 535.39 | 582.43 |
| Balanced / immediate leader / P96 | 1594.95 | 43.55 | 0.00 | 545.44 | 580.70 |
| Balanced / immediate leader / P64 | 1609.25 | 42.32 | 0.00 | 568.78 | 587.42 |
| Random / separate ready / P96 | 1632.45 | 39.84 | 12.64 | 587.20 | 584.13 |
| Random / immediate leader / P96 | 1639.22 | 43.48 | 0.00 | 601.67 | 585.55 |

The standalone ready interval is zero because the node is absent in all four
captured graphs. Its necessary synchronization now runs inside GEMM1. Against
separate-ready P96, the leader version has E2E +0.40% balanced and +0.41% random;
these small single-run differences do not establish a stable regression or gain.
GEMM1 is slower by 10.05 us balanced and 14.47 us random; compact-plan also does
more setup work. P64 did not help, so the reproduction script retains P96.

Full Flash four-GPU correctness passed for P96 and P64, with changing graph
inputs, random/hot/local-only/unequal-tail routing, exact payload/scale/map checks,
and a delayed-source probe proving a nonleader can publish a local tile while
the leader waits. The probe uses runtime pointers and bounded waits. All tests
and timing jobs in this final leader iteration exited successfully without a
boot change. No stable E2E improvement is claimed; default remains off.

Current report, historical variants, per-run symbols/source/CSV snapshots and
native/HIP event results: `/workspace/gpubench/s1f_ready_first_20261119/REPORT.md`.

## Histogram-first initialization follow-up

The retained implementation now submits the histogram before initializing local
metadata, queue heads and epochs. Flash uses the tagged dense histogram path:
its asynchronous TDM transfer can remain in flight while independent GMEM setup
runs. The setup does not write the histogram's LDS source. Its store retirement
and TG barrier still precede the live metadata scan; histogram tag checks and
TDM completion before LDS reuse remain unchanged. Sparse and non-tag fallback
paths retain their original immediate completion waits. Quantization is unchanged.

This is a relocation of one setup block, without a new kernel or feature flag.
The producer-leader and consumer implementations are unchanged. The comparison
below has READY_FIRST=1 throughout and only switches initialization order.
Same EP4/tpr16384/P96 configuration and timing methodology as above; all values
are microseconds per layer. Execution order for the balanced comparison was
histogram-first run1, refreshed init-first control, histogram-first run2.

| Routing / order | Plan + quant | GEMM1 + producer | GEMM2 + scatter | Native E2E |
|---|---:|---:|---:|---:|
| Balanced / init first, refreshed | 43.24 | 545.39 | 580.02 | 1594.75 |
| Balanced / histogram first, run 1 | 42.58 | 553.19 | 591.37 | 1596.64 |
| Balanced / histogram first, run 2 | 42.87 | 549.93 | 585.40 | 1594.79 |
| Random / init first, saved | 43.48 | 601.67 | 585.55 | 1639.22 |
| Random / histogram first | 43.65 | 601.03 | 586.03 | 1642.06 |

Balanced plan is0.37–0.66us lower than the refreshed control; random plan is0.17us
higher than its saved control. This limited sample supports at most a small,
workload-dependent change, not a stable overall speedup. Native E2E is effectively
unchanged in the balanced repeat; no E2E improvement is claimed. The reorder is
retained and READY_FIRST remains opt-in/default0. Earlier leader timing tables
above predate this reorder and remain historical comparisons.

Validation: ipflegacy_p96_check_hist_overlap_r1 passed the full four-GPU six-routing
check and delayed-source local-progress probe. All four follow-up timing jobs
exited0 with unchanged boot IDs. Every opt-in rank graph still has zero ready
nodes, eight plan nodes and eight nodes for each GEMM. Three review passes and
paired-source switches are recorded in REVIEW.md. Exact pre/post source is in
compact_plan_before_hist_overlap.py / compact_plan_hist_overlap.py; per-run full
source diffs, environments, logs and event timing remain under the run directories.


## Completion-driven S1F work queue (experimental)

Set `AITER_MOE_S1F_READY_QUEUE=1` together with the existing pull S1F flags and
`AITER_MOE_S1F_READY_FIRST=1`. The queue flag defaults to0. This mode requires
plan alignment equal to GEMM1 tile M and independent GEMM consumers (cluster1x1);
producer affinity placement may remain enabled. Existing resident-worker guards
still apply. GEMM tiling, copies/shuffle, GEMM2 and producer-to-consumer transition
are unchanged.

Previously, consumers claimed a predetermined M/N task and could then wait on
that specific A tile even when other A tiles were ready. In ready-queue mode,
each live M64 producer wave retires its stores, publishes its original flag and
increments an agent AcqRel completion counter for its containing A tile. The
last live part appends M+1 with a release store. Padding parts do not participate.
Consumers acquire this entry and independently claim its N tasks. For Flash
I2304/N256, a single completed A256 exposes18 tasks;18 distinct simultaneously
available workers are not guaranteed. M order is determined by completion, not
by a static task-index formula. N index remains ticket%18.

The append-only queue uses one record per A tile, not18 redundant task records.
Its storage follows the256 affinity heads:64 i32 header words plus two arrays
of ceil(compact_cap/64) words, for completion counters and queue entries. No new
LDS is allocated. Compact-plan clears it before each S1F launch/replay. Consumers
can reserve future queue entries and wait for publication; they do not select an
unready M tile in advance. A briefly unpublished append reservation can delay a
claim even if a later entry has published. Producers never wait for consumers.
The existing A-ready checks and next-task claim/metadata-prefetch hooks remain.

Full EP4/16384 tokens per rank correctness passed six changing graph routing
cases (random, hot, rank-local, unequal tails), exact payload/scale/map checks,
queue permutation and completion-count checks, and the delayed-source progress
probe. No crash or boot change was observed. Three review passes preceded launch.

Balanced Flash E384/topk6/H5120/I2304, P96/C160,8layers, warmup5/iters32;
GEMM1 tile256x256x256,4buffers,2x2waves. Same CSV and other settings in both modes.
Numbers are microseconds/layer. E2E is native graph timing, averaged rank medians
and divided by8; individual stages are means from a separate HIP graph-event
pass across4ranks,8layers and32replays. Stages must not be summed into native E2E.
Execution order: on1, off1, off2, on2.

| Ready queue | Run | Plan + quant | GEMM1 + producer | GEMM2 + scatter | Native E2E |
|---|---:|---:|---:|---:|---:|
| Off |1|43.12|551.81|598.81|1606.56|
| Off |2|42.90|548.75|591.99|1602.19|
| On |1|41.79|568.21|578.63|1593.45|
| On |2|43.82|568.93|591.57|1617.59|

Mean GEMM1 increases from550.28 to568.57us (+3.32%). Native E2E means are1604.37
and1605.52us (+0.07%), with overlapping run variation. No performance improvement
is established. Added completion atomics and one shared claim head, and losing
consumer-local A reuse, are plausible costs; these timings do not isolate their
individual contributions. Queue mode remains optional.

Reproduce queue-on/off with BENCH_READY_QUEUE=1/0 and a unique label:

```bash
BENCH_IPF=legacy BENCH_PRODUCERS=96 BENCH_READY_QUEUE=1 BENCH_BALANCED=true bash /workspace/gpubench/s1f_ready_queue_20261119/run.sh affinity queue_on_unique
```

Reports, exact commands/environment/source, correctness logs and stage timings:
`/workspace/gpubench/s1f_ready_queue_20261119/REPORT.md`.


## Cluster-local signal/wait experiment

`AITER_MOE_S1F_CLUSTER_LOCAL=1` selects a separate pull S1F scheduler, with
`AITER_MOE_S1F_READY_FIRST=1` and `AITER_MOE_S1F_READY_QUEUE=0`. Default is0.
It uses hardware cluster barriers for producer/consumer rendezvous and no global
atomic RMW in GEMM1 (verified in all four rank code objects). Cross-device source
readiness still uses release/acquire global loads/stores. Compact-plan is unchanged.

At P96/256 workers, each4x2 physical cluster has3 producers and5 initial consumers.
A round prepares3 A256 tiles; consumers prefetch B before waiting for all A data.
Producers then join consumption of54 M/N tasks. A second collective barrier
finishes the round before the next begins. M ownership is fixed per cluster;
there is no work stealing or original next-item metadata-prefetch hook in this
prototype. GEMM tiling, gather/shuffle, A preshuffle and instruction prefetch are
retained. The new optional GEMM prologue hook lets all waves wait after B issues
and before A issues. Requirements: aligned plan/GEMM M, WPT1 or2, four-wave GEMM,
resident workers and a complete mixed-role affinity cluster.

Four-GPU barrier probe and full Flash six-routing graph correctness passed.
Balanced Flash EP4 at16384 tokens/rank (65536 global), P96,8layers:

| Variant | Plan + quant | GEMM1 + producer | GEMM2 + scatter | Native E2E/layer |
|---|---:|---:|---:|---:|
| Affinity, saved two-run mean | 43.01 | 550.28 | 595.40 | 1604.37 |
| Global ready queue, saved two-run mean | 42.81 | 568.57 | 585.10 | 1605.52 |
| Cluster signal/wait, new single run | 44.11 | 722.69 | 585.44 | 1763.44 |

Baselines are saved two-run means; cluster mode is one new run. Stages use a
separate HIP graph-event pass; E2E uses native timing. GEMM1 regresses31.3%, E2E9.9%.
Collective tile readiness, dispatch/compute round boundaries, fixed ownership
and loss of next-item metadata overlap are structural costs. Hardware barrier
latency alone was not isolated. No private scratch allocation or VGPR spilling
was found; compiler-reported SGPR spilling is lower than the cached baseline.

Full protocol, validation, assembly evidence and reproduction:
`/workspace/gpubench/s1f_cluster_signal_20261119/REPORT.md`.

## Four-TG cooperative S1F producers (experimental)

`AITER_MOE_S1F_PRODUCER_GROUP=4` makes four producer TGs cooperate on each
A256. The default is1, retaining the existing producer. This option requires
pull S1F, GEMM1 M256, a producer count divisible by4, and a resident worker count
divisible by4 and at most min(256, device CU count). Producer count must be
positive and smaller than worker count. It is incompatible with ready-queue and
cluster-local scheduling. Both external-ready and ready-first paths are supported.

Hardware launch clusters are2x2 with homogeneous initial roles: the first P TGs
form P/4 complete producer clusters. Each TG copies one M64; each of its four
waves copies one M16 using the existing gather/shuffle helper. Every wave retires
its payload/scale stores, followed by a release fence, TG barrier and cluster
signal/wait. After an acquire fence, TG0 publishes the existing four M64 ready
flags for live parts. Empty/tail members still join the collective. This adds no
global completion atomic. Finished producers join independent consumers.

GEMM consumers remain1x1: no multicast or consumer cluster barrier is enabled.
The hardware launch attribute also applies to consumer TGs, but does not make
their computation collective. Original task-claim, stealing, metadata/ready
prefetch, same-A reuse and epilogue hooks remain. There are P/4 logical task
queues corresponding to producer groups. This changes physical placement from
the old mixed-role affinity clusters; it does not retain their producer/consumer
cache affinity. Compact-plan and GEMM2 algorithms are unchanged.

Validation passed on four GPUs at16384 tokens/rank (65536 global), including
both ready paths, changing graph inputs, hot/rank-local routing, unequal tails,
exact payload/scale/map checks and output comparison with push. A prior small
P100 check also passed, verifying that P must be divisible by4, not necessarily8.

Balanced Flash E384/topk6/H5120/I2304, P96/C160,8layers, warmup5/iters32,
legacy instruction prefetch, BTH0, external-ready; same CSV in both variants.
GEMM1 and GEMM2 use256x256x256,2x2waves,4buffers. GEMM1 uses WPT2; GEMM2 uses
WPT1 and cluster1x4. Order: baseline1, group4_1, group4_2, baseline2.

| Producer | Run | Plan + quant | Ready | GEMM1 + producer | GEMM2 + scatter | Native E2E/layer |
|---|---:|---:|---:|---:|---:|---:|
| Original affinity |1|39.93|12.97|542.01|596.20|1600.90|
| Original affinity |2|39.44|13.06|534.46|586.00|1588.53|
| Four TGs per A256 |1|37.33|12.56|578.22|586.22|1607.94|
| Four TGs per A256 |2|38.12|12.63|578.30|585.61|1605.46|

All times are microseconds. E2E averages native rank medians and divides by8;
stages average a separate HIP graph-event pass across four ranks, eight layers
and32 replays. Do not sum stages to reconstruct native E2E. Two-run means:
GEMM1 increases538.24 to578.26us (+7.44%); E2E increases1594.72 to1606.70us
(+0.75%). Native arithmetic-mean E2E also regresses1594.96 to1615.07us (+1.26%);
the first group4 run contains a high-latency replay. No overall gain is established.

The earlier isolated remote-A256 microbenchmark reduced tile completion latency
from87581 to22292cycles. It did not measure the producer inside full MoE. At
fixed P96, grouping reduces concurrently produced A256s from at most96 to24;
lower tile latency does not imply increased aggregate dispatch throughput.
Changed cache affinity, queue distribution and collective arrival imbalance can
also cost performance; their individual contributions have not been isolated.
The group4 mode remains opt-in, and group1 remains the performance default.

Reproduce native timing (use group1 for the original affinity baseline):

```bash
env -u AITER_FLYDSL_INST_PREFETCH AITER_MOE_S1F_AFFINITY=1 AITER_MOE_S1F_PRODUCER_GROUP=4 AITER_MOE_S1F_READY_FIRST=0 DS41_PRODUCERS=96 bash run_ds41_flash_best.sh
```

Exact source/environment snapshots, four-run stage timings, correctness logs,
analysis script and reproduction commands:
`/workspace/gpubench/s1f_producer_group4_20261119/REPORT.md`.

The subsequent group4 producer-ratio sweep kept256 workers and the same full
Flash workload. P64 and P96 are two-run means; other ratios have one run.

| Producer TGs | Initial consumer TGs | GEMM1 + producer | GEMM2 + scatter | Combine sync | Native E2E/layer |
|---:|---:|---:|---:|---:|---:|
|32|224|956.96|583.47|32.78|1991.53|
|48|208|690.17|590.29|19.77|1721.11|
|64|192|567.31|590.44|22.44|1609.15|
|80|176|567.47|591.17|28.92|1616.49|
|96|160|578.26|585.92|14.99|1606.70|
|128|128|584.05|581.55|35.23|1625.95|

Units and timing scopes are unchanged. Combine sync is only
`ep_combine_fused_sync_0`. P64 lowers GEMM1 by1.89% against P96, but increased
GEMM2/combination wait intervals offset the saving. Native E2E run ranges overlap;
there is no established overall gain. P64-P80 is the best observed GEMM1 region,
while original affinity P96 remains the overall default. Group4 P64 passed the
full four-GPU correctness test for both ready paths. Use `DS41_PRODUCERS=64` in
the reproduction command above to select it. Full ratio data and per-run
variation are in the report's producer/consumer sweep section.

### One ready flag per pull S1F M256

`AITER_MOE_S1F_TILE_READY=1` (default) enables one generation flag per M256
for the one-TG pull producer. Each of its four waves retires
its M64 payload/scale stores; a TG barrier and release/acquire ordering precede
wave 0's publication. Empty/tail waves participate in the same barrier. The
consumer loads one local-source dword and retains next-task readiness prefetch
and same-A reuse. The flag reuses the first M64 slot; metadata and arena sizes
are unchanged. Kernel symbols contain `_readyM256` when selected.

With `AITER_MOE_S1F_READY_FIRST=1`, producer slot 0 publishes local plan/quant
readiness, collects remote source readiness and publishes the local `all_ready`
generation before returning to production. Other producers can publish fully
local M256 tiles before `all_ready`; mixed/remote tiles publish after phase 1.
No standalone ready kernel is used in this mode.

Set `AITER_MOE_S1F_TILE_READY=0` to compare the old per-M64 protocol. M64/M128
GEMMs, ready-queue, cluster-local and four-TG producer experiments retain their
existing protocols. The effective config logs `tile_ready` and keys the JIT
cache on it. This flag changes shuffled-A completion, not the source-ready
synchronization required before remote pulls.

### Experimental compact-plan packing and M256 source-rank shuffle

Read [README_dispatch_pull.md](../../../../../README_dispatch_pull.md) before
continuing this experiment on another host. It contains the full code map,
layout, readiness protocol, reproducible commands, cycle findings and pending work.

`AITER_MEGA_PULL_PLAN_PACK=1` quantizes and packs source frames in the existing
compact-plan; S1F producers use ordinary TDM loads. Add
`AITER_MEGA_PULL_SRC_SHUFFLE=1` to align every `(expert, source_rank)` segment
to M256. **Every valid row in an M256 then belongs to the same source rank,
including normal routing. Padding rows are not tokens and have route=-1.**
Both new switches default off. This first version requires external-ready S1F,
one-TG producers and M256. GEMM2 code is unchanged; its existing negative-route
scatter mask discards padding.

Status: the earlier packed-load prototype passed EP4 correctness at257 and16384
per rank; the added source-alignment/padding code is not yet GPU-validated.
No packed or source-shuffle speedup has been measured. The user requested this
checkpoint before GPU access became available. First validate, then establish a
GEMM1 improvement; only afterward experiment on GEMM2.

Three source-shuffle reviews before GPU launch: (1) capacity covers all source
segment tails, source owners alone invalidate their padding, packed pointers
are64-bit; (2) planner-built live metadata survives external-ready, quant waits
for released route-map generations, existing source/A-ready protocols remain;
(3) tests check source uniqueness, exact route coverage, live-row metadata,
outputs and payload/scales/weights under changing and unequal routing.
