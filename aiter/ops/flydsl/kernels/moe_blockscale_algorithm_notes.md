# Blockscale MoE Algorithm Notes

This note records algorithm-level experiments for
`moe_blockscale_2stage.py`.  It is intentionally separate from tuned CSV
entries: a design can exist before it is fast enough to dispatch.

## Stage1 DMA Variant

The implemented `optdma` stage1 variant keeps the existing two-buffer
compute pipeline and only changes the X tile producer:

1. Issue `raw_ptr_buffer_load_lds` from X global memory directly into the
   XOR16 LDS layout.
2. Keep gate/up B loads and blockscale scale loads in the existing
   `do_one_stage` prefetch group.
3. Wait with `s_waitcnt(0)` before the CTA barrier that makes the producer
   buffer visible to the next compute stage.
4. Keep the split-K semaphore and fp32 partial accumulation path unchanged.

The scheduler accounting treats DMA as VMEM reads with zero DS writes.  This
matches the existing stage2 DMA path and isolates the algorithm change from
the epilogue and split-K correctness contract.

## Stage2 M-Persistence Probe

A simple CTA-local M persistence prototype (`persist_m=2/4`) was tested for
DS V3.2 TP4 large-token rows by grouping adjacent sorted-M tiles in one CTA.
The code path was removed after measurement instead of being kept as an
unprofitable opt-in variant:

```text
atomic pm2:      8192=2499.85us, 32768=9153.04us
atomic pm4:      8192=2429.48us, 32768=8599.93us
dma atomic pm2:  8192=2462.54us, 32768=9162.35us
reduce pm2:      8192=2458.29us, 32768=8271.84us
reduce pm4:      8192=2524.79us, 32768=8323.43us
baseline:        8192=~2265us,  32768=~8210us
```

The result suggests these shapes need stage2 CTA parallelism more than they
benefit from L2 reuse of neighboring M tiles.  Future stage2 work should focus
on reducing atomic/reduction overhead or improving N/K scheduling rather than
serializing M tiles inside a CTA.

## Stage1 Three-Stage Ring Buffer Result

The first `optms3` prototype implements the planned X-only three-stage ring
buffer for stage1.  It keeps B and blockscale scale prefetch in registers and
adds a third LDS X stage so the loop can prefetch `k+2` while computing `k`.

GPU4 DS V3.2 TP4 results:

```text
kb2 baseline: 8192=2288.94us, 32768=8101.02us
kb2 optms3:  8192=2286.98us, 32768=8110.76us
kb1 optms3:  8192=2418.65us, 32768=8370.65us
```

The full applicable-token sweep passed correctness, but small/mid tokens were
mostly slower and large tokens were within noise.  This suggests the current
stage1 bottleneck is not first-X-tile LDS latency once split-K is enabled; the
next higher-value step is slice-K / warp-level K partitioning rather than more
X buffering.

## Three-Stage Ring Buffer Prototype

The safe MoE port of PR #3279's multi-stage idea should start with X-only
staging.  B and blockscale scales are still expert-dependent and should remain
in the current `do_one_stage` prefetch/carry state until the X path proves
useful.

Target layout:

```text
lds_x_stage0 = 0
lds_x_stage1 = tile_m * lds_stride
lds_x_stage2 = 2 * tile_m * lds_stride
```

For CShuffle kernels, `lds_out` may still alias the X allocation after the
mainloop completes, so the LDS allocation must be:

```text
max(3 * tile_m * lds_stride * elem_bytes, lds_out_bytes)
```

Target dataflow:

```text
prefetch X[k0] -> stage0
wait, barrier
prefetch X[k1] -> stage1
load scales/B[k0]

loop:
  prefetch X[k+2] -> free_stage
  load scales/B[k+1] into loop-carried registers
  compute k using compute_stage
  wait, barrier
  rotate compute_stage, next_stage, free_stage
```

Important constraints:

- The compute stage must never be overwritten until all LDS reads for that K
  tile have completed.
- `a0_prefetch` must be taken only after the matching DMA wait/barrier.
- The first prototype should be an opt-in `opt_variant` and should not change
  the current `full` or `dma` behavior.
- The existing split-K semaphore must remain after the mainloop, because
  cross-CTA K slices still reduce through global atomics.

## Intra-CTA Slice-K Design

PR #3279's `block_k_warps` splits a CTA's K tile among warp groups and reduces
their partial sums inside the CTA before the global epilogue.  For blockscale
MoE the equivalent design is:

1. Split each `tile_k` into `block_k_warps` slices.
2. Assign warp groups by `(warp_mn, warp_k)`, where `warp_k` owns a contiguous
   K slice aligned to `scale_block_k`.
3. Each K slice computes private gate/up accumulators in fp32.
4. Store those private accumulators to LDS as:

```text
partial_gate[block_k_warps, tile_m, tile_n]
partial_up[block_k_warps, tile_m, tile_n]
```

5. Reduce the `block_k_warps` dimension in LDS inside the CTA.
6. Feed the reduced gate/up tile into the existing CShuffle epilogue.
7. If `k_batch > 1`, the reduced CTA tile still participates in the existing
   global split-K semaphore/atomic path.

Correctness requirements:

- Slice boundaries must be multiples of `scale_block_k=128`, otherwise a
  slice would need to share one blockscale factor with a neighboring slice.
- The reduce must happen before SiLU.  Applying SiLU per slice would be wrong
  because `silu(sum(gate_slices)) * sum(up_slices)` is not separable.
- The first implementation should target stage1 only.  Stage2 has a smaller K
  dimension and already has a tuned B-split path, so slice-K is less likely to
  pay for its LDS reduction overhead there.

Risk:

- LDS pressure grows with `block_k_warps * 2 * tile_m * tile_n * 4B` before
  reduction.  For the common `64x128` stage1 tile this is already 64 KiB at
  `block_k_warps=2`, before X LDS and `lds_tid`, so the first practical slice-K
  candidate likely needs smaller `tile_m` or `tile_n`.

## Stage1 Slice-K Probe

The first `optslice2` prototype is intentionally narrow:

```text
stage1 only
gfx950 blockscale MFMA only
tile_m=16, tile_n=128, tile_k=256
k_batch=1
block_k_warps=2
```

Wave mapping changes from four N waves to `(n_wave=2, k_wave=2)`.  Each K wave
computes one 128-wide blockscale slice, writes fp32 gate/up fragments to LDS,
then the `k_wave=0` group reduces the second slice before the direct epilogue.
This keeps the SiLU contract correct because reduction happens before
`silu(gate) * up`.

GPU4 DS V3.2 TP4 result:

```text
baseline t16x128x256: token32=347.10us, token128=500.28us
optslice2:            token32=349.98us, token128=498.40us
```

Correctness matches the existing warning profile, but the speedup is not stable
enough to write back.  The likely remaining overhead is that the first prototype
still prefetches the full B/scale tile before each K wave consumes only one
128-wide slice.  A follow-up slice-K candidate should make B/scale prefetch
slice-aware before testing broader token coverage.

## Bottleneck-First Roofline Pass

MI355X peak specs used for roofline reasoning:

```text
HBM bandwidth: 8 TB/s
FP8 matrix peak: 5 PFLOPS
ridge point: ~625 FLOP/byte
```

For the profiled DS V3.2 TP4 blockscale shape
(`model_dim=7168, inter_dim=256, topk=8, tile_m=16, tile_n=128,
tile_k=256`), the stage1 logical work is only about 0.0587 GFLOP per
input token.  Measured stage1 throughput on GPU4 was far below the 5 PFLOP
compute roof:

```text
token32:    ~15.5 TFLOP/s  (~0.31% of FP8 peak)
token128:   ~46.9 TFLOP/s  (~0.94% of FP8 peak)
token512:  ~144.1 TFLOP/s  (~2.88% of FP8 peak)
token8192: ~513.1 TFLOP/s  (~10.3% of FP8 peak)
token32768:~593.0 TFLOP/s  (~11.9% of FP8 peak)
```

Profiler counters support the memory-bound interpretation, especially for
small and mid tokens:

```text
stage1 token512:   SQ_INSTS_VMEM_RD=2.25M,  SQ_INSTS_VALU_MFMA_F8=0.66M
stage1 token8192:  SQ_INSTS_VMEM_RD=25.6M,  SQ_INSTS_VALU_MFMA_F8=7.56M
stage1 token32768: SQ_INSTS_VMEM_RD=100.1M, SQ_INSTS_VALU_MFMA_F8=29.6M
stage1 LDS bank conflict: 0
```

The important takeaway is that X-only multi-stage is not enough.  The kernel
must either reduce B/scale traffic per useful MFMA or increase effective B
reuse across sorted-M work for the same expert.  PR #3279's merged HGEMM work
is best understood as streaming-K plus mixed slice/split-K policy; our first
`optms3` port only covered X-side staging, so its result does not invalidate
the PR direction.

## Stage1 B-Split Scheduling Probe

A stage1-only B-split experiment mirrored the stage2 pattern:

```text
load B low half for next tile in the prefetch group
load B high half inside compute for the current tile
```

This was correct but slower on GPU4:

```text
token32:    full=113.89us, bsplit=118.70us, speedup=0.96x
token128:   full=154.77us, bsplit=156.17us, speedup=0.99x
token512:   full=202.40us, bsplit=205.50us, speedup=0.98x
token8192:  full=931.69us, bsplit=997.50us, speedup=0.93x
token32768: full=3228.56us, bsplit=3473.63us, speedup=0.93x
```

The code path was removed after measurement.  Moving B loads later reduces
some live range pressure but does not reduce B bytes or improve cross-CTA B
reuse, so the extra scheduling/register overhead dominates.

## Stage1 Mixed Split-K Policy Probe

Directly forcing split-K everywhere is not a good policy: it adds a global
atomic epilogue plus a post `silu_and_mul` kernel, and that overhead becomes
dominant once the normal `k_batch=1` launch has enough M work.  However,
fixed-shape DeepSeek V3.2 TP4 stage1 does benefit from split-K at small token
counts because `k_batch=7` exposes more K-parallel CTAs while keeping four
`tile_k=256` tiles per split.

Measured on gfx950 with `tile_m=16, tile_n=128, tile_k=256, waves_per_eu=3`:

```text
token32:   k_batch=1 227.34us, k_batch=7 169.02us
token128:  k_batch=1 265.06us, k_batch=7 251.01us
token512:  k_batch=1 365.32us, k_batch=7 354.55us
token1024: k_batch=1 596.96us, k_batch=7 581.29us
token2048: k_batch=1 1082.41us, k_batch=7 1007.31us
token4096: k_batch=1 1665.59us, k_batch=7 1690.39us
token8192: k_batch=1 2921.19us, k_batch=7 3072.24us
```

Do not encode this as a runtime `skauto` policy in the kernel wrapper.  The
tuner should emit explicit configs so the selected launch is transparent:

```text
small-token rows: choose an explicit k_batch=7 config when it wins
large-token rows: choose an explicit k_batch=1 config when split-K loses
```

This keeps the kernel body unchanged and avoids hiding the true config behind
a host-side heuristic.  Correctness against the bf16 torch reference stayed
within blockscale noise on small tokens:

```text
token32:  max_abs=0.00781, mean_abs=0.00000
token128: max_abs=0.00781, mean_abs=0.00001
token512: max_abs=0.01562, mean_abs=0.00001
```

## Stage1 LDS Scale-X Cache

The next real bottleneck after removing experimental variants was duplicated
stage1 activation-scale traffic. Each CTA needs only `sb_per_tile * tile_m`
unique `scale_x` values for a K tile, but the old per-lane path loaded the same
row scale repeatedly across lane groups and N waves. For the DS V3.2 TP4
default shape this means roughly 32 unique scale loads were expanded into
hundreds of VMEM instructions per CTA.

The production path now caches `scale_x` in a small LDS scratch. The first
synchronized version used one scratch buffer; the current version uses a
ping-pong scale scratch aligned with the existing X tile ping-pong pipeline:

```text
first sb_per_tile*tile_m threads: load one scale_x each into LDS
all lanes: read per-row scale from LDS before blockscale MFMA accumulation
next K tile writes the other scale scratch slot
```

This does not add a tuning knob or a runtime policy. It directly reduces scale
VMEM traffic while preserving the existing two-stage ping-pong pipeline.

Measured stage1 after the synchronized cache on gfx950 with
`tile_m=16, tile_n=128, tile_k=256, waves_per_eu=3`:

```text
token32:   k_batch=1 220.34us, k_batch=7 174.49us, k7 vs k1 +26.3%
token128:  k_batch=1 256.97us, k_batch=7 252.85us, k7 vs k1 +1.6%
token512:  k_batch=1 357.69us, k_batch=7 351.34us, k7 vs k1 +1.8%
token2048: k_batch=1 1062.57us, k_batch=7 984.37us, k7 vs k1 +7.9%
token4096: k_batch=1 1657.94us, k_batch=7 1699.51us, k7 vs k1 -2.5%
```

The ping-pong LDS scale scratch removes the end-of-load barrier while avoiding
the race in the unsynchronized single-buffer prototype. It especially helps the
explicit split-K path:

```text
token32:    k_batch=1 221.15us,  k_batch=7 162.35us, k7 +7.5% vs sync-cache k7
token128:   k_batch=1 263.71us,  k_batch=7 238.79us, k7 +5.9% vs sync-cache k7
token512:   k_batch=1 364.27us,  k_batch=7 323.69us, k7 +8.5% vs sync-cache k7
token2048:  k_batch=1 1091.26us, k_batch=7 897.00us, k7 +9.7% vs sync-cache k7
token4096:  k_batch=1 1698.51us, k_batch=7 1488.64us, k7 +14.2% vs sync-cache k7
token8192:  k_batch=1 3041.04us, k_batch=7 2763.65us, k7 +11.2% vs sync-cache k7
token32768: k_batch=1 11195.62us, k_batch=7 10298.18us
```

All measured points passed the bf16 stage1 reference check with `atol=0.5,
rtol=0.1`. This makes the PR3279-style resource-rotation idea productizable for
stage1 without reintroducing `opt_variant`: the tuning surface remains the
explicit `k_batch`.

Stage2 has a similar duplicated-scale-load pattern, but its torch-reference
delta is already near the tuner tolerance boundary. A synchronized stage2 cache
trial did not provide a clean enough correctness/performance story: torch
reference error did not materially worsen, but output identity against the
current FlyDSL baseline was not proven and small-token latency was flat/slower.
Keep stage2 scale caching out of the production path until that precision
baseline is isolated separately.

## Stage1 Multi-Stage Follow-Up

The generic `stages>2` ring is correctness-clean, but the original form kept a
full future B tile in loop state. That increases VGPR/state pressure and did not
match PR3279's LDS-resident A/B pipeline.

Async X DMA was isolated rather than skipped:

```text
raw X gmem->LDS DMA, scheduler off: correctness passes
single scheduler groups (dsrd/mfma/vmem/dswr): correctness passes
full combined scheduler groups: LLVM SmallVector assertion during compile
```

For now the safe async path keeps the scheduler hint off. On `tile_k=256,
k_batch=7` it is correctness-clean and roughly neutral/slightly better than the
non-async path, but `tile_k=128` remains faster overall.

A B-to-LDS ring prototype was tested for the feasible LDS budget point
`tile_n=64,tile_k=128,stages=3,b_lds=True`. It also passed correctness, but
occupancy dropped from 3 to 2 and latency regressed:

```text
token32:   s2=109.55us, s3_reg=107.53us, s3_blds=194.27us
token128:  s2=160.00us, s3_reg=163.16us, s3_blds=213.39us
token512:  s2=230.07us, s3_reg=228.15us, s3_blds=296.47us
token2048: s2=406.14us, s3_reg=412.54us, s3_blds=759.37us
```

The useful follow-up tuning space is `tile_n=64` with the register pipeline:

```text
tile_n=64,tile_k=128,k_batch=1:
token32=110.00us, token128=160.12us, token512=226.30us, token2048=408.96us

tile_n=128,tile_k=128,k_batch=1:
token32=117.62us, token128=163.26us, token512=205.19us, token2048=560.27us
```

So `tile_n=64` should be part of offline tuning. A follow-up sweep against
`tile_n=128,tile_k=128` showed that it is token-shape dependent rather than a
global replacement:

```text
token32:   n64 107.42us (kb=1,s=3,wpe=3) vs n128 110.53us (kb=7,s=2,wpe=3)
token128:  n64 160.14us (kb=1,s=2,wpe=2) vs n128 148.96us (kb=1,s=2,wpe=2)
token512:  n64 218.84us (kb=2,s=3,wpe=4) vs n128 202.69us (kb=1,s=2,wpe=2)
token2048: n64 408.26us (kb=1,s=2,wpe=2) vs n128 547.44us (kb=1,s=3,wpe=2)
token8192: n64 1120.00us (kb=1,s=2,wpe=2) vs n128 983.98us (kb=1,s=2,wpe=2)
```

The full-B LDS prototype was removed from the production tuning surface after
this measurement. A truly profitable B-to-LDS path likely needs a lighter slice
than full gate+up K128 per stage, for example separate gate/up staging or
another layout that avoids the occupancy drop.

## GFX950 Scheduler Count Correction

The blockscale gfx950 path uses `mfma_scale_f32_16x16x128_f8f6f4`, but the
scheduler annotations still counted the older K64 MFMA shape. That over-counted
MFMA groups and under-counted DS reads for `tile_k=256`. Stage2 also inherited
the stage1 scale-cache VMEM count even though stage2 still loads `scale_x`
directly per `(sb, mi, ii)`.

The production fix updates only scheduler counts:

```text
stage1 gfx950 DS reads:  2 * sb_per_tile * m_repeat
stage1 gfx950 MFMA:      2 * sb_per_tile * m_repeat * num_acc_n
stage2 gfx950 DS reads:  2 * sb_per_tile * m_repeat
stage2 gfx950 MFMA:      1 * sb_per_tile * m_repeat * num_acc_n
stage2 scale_x VMEM:     sb_per_tile * m_repeat * 4
```

This does not change math, memory layout, launch shape, or tuning policy. A
focused stage1 microbenchmark on gfx950 remained correctness-clean and was
neutral to slightly faster:

```text
token512,  tile_n=64:  229.73us -> 227.79us
token512,  tile_n=128: 206.39us -> 207.23us
token2048, tile_n=64:  402.47us -> 402.04us
token2048, tile_n=128: 558.45us -> 555.86us
```

Production `--run_config` validation passed on representative DS V3.2 TP4
FlyDSL rows for tokens `128, 512, 8192, 16384` with the existing tuned CSV.

Two other product candidates were measured and rejected:

```text
bf16 stage1 CShuffle epilogue:
  correctness-clean, but tile_n=64 and token512 regressed; only token2048/tile_n=128
  showed a small win.

M-major stage1 launch order:
  token512/tile_n=64 improved slightly, but token2048 regressed heavily.
```

Keep both out of production unless they are exposed later as explicit tuned
config dimensions with stable per-shape wins.

## DS V3.2 TP4 Retune Checkpoint, 2026-06-01

Comparison baseline is the main-branch tuned config for
`cu=256, model_dim=7168, inter_dim=512, expert=257, topk=9`. The retuned
FlyDSL config improves the common 14-token kernel geomean by `+5.95%`, but the
model-level DeepSeek V3.2 serving benchmark is still effectively flat.

Kernel latency comparison:

```text
token     main tuned us   retuned us   speedup
1              56.84        40.42      +40.64%
2              62.72        54.69      +14.69%
4              78.65        83.32       -5.60%
8             128.17       125.84       +1.86%
16            211.27       211.30       -0.01%
32            328.35       327.93       +0.13%
64            418.70       420.97       -0.54%
128           472.00       462.90       +1.96%
256           515.70       497.79       +3.60%
512           540.50       528.00       +2.37%
1024          673.68       590.33      +14.12%
2048          873.59       842.32       +3.71%
4096         1294.01      1245.53       +3.89%
8192         2305.85      2107.27       +9.42%
```

DeepSeek V3.2 serving rerun used GPUs `0,1,2,3`, `ISL=1000`, `OSL=100`,
`num_prompts=8*concurrency`, and `num_warmups=concurrency`. The service was
started with `PYTHONPATH=/workdir/aiter_test:...` and
`AITER_CONFIG_FMOE=/workdir/aiter_test/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_ds_v3.csv`.

```text
concurrency   baseline total tok/s   retuned total tok/s   gain
4                    2596.07              2614.41          +0.71%
8                    4280.34              4110.11          -3.98%
16                   5779.38              5989.69          +3.64%
32                   8261.10              8303.30          +0.51%
```

The E2E geomean is only `+0.18%`. The `concurrency=8` regression is the most
important next target: the tuned kernel gains are not reliably translating to
serving throughput yet, so further work should prioritize end-to-end MoE
scheduling and candidate selection rather than only single-shape kernel latency.
