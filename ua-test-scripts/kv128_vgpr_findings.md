# CK-UA fp8 prefill: VGPR analysis for kBlockM=256, kv128

Goal: understand why CK unified-attention (prefill_d128 fp8) only fits KV tile
= 64 while the ASM fmha_sage kernel fits KV tile = 128 at the same BlockM=256,
and find the VGPR-pressure cut that unlocks kv128.

All numbers are the gfx950 device kernel resource footer
(`.vgpr_count` / `.vgpr_spill_count`) from a single-TU compile of
`unified_attention_d128_fp8_mask.cpp` (instance == the hot prefill dispatch),
captured with `ua-test-scripts/measure_vgpr.sh`. 256 is the hard per-wave VGPR
ceiling on gfx950; anything above spills to scratch.

## Headline measurements

| config | VGPR | spill | LDS | scratch |
|---|---|---|---|---|
| kv64 (production) | 214 | 0 | 64 KB | 0 |
| kv128 naive | 256 | 173 | 128 KB | 304 B |
| kv128 + shared sp_compute (E2) | 256 | **126** | 128 KB | 244 B |
| kv128 + union k/v (E3) | 256 | 173 | 128 KB | 304 B |
| kv128 + E2 + E3 | 256 | 126 | 128 KB | 244 B |
| kv128 + E2 + CONDITIONAL_RESCALE=0 | 256 | 128 | 128 KB | 344 B |
| kv128 + wide-MMA off | 256 | 172 | 128 KB | 332 B |
| kv128 nopage | 256 | 156 | 128 KB | 288 B |
| kv128 nopage + E2 | 256 | 113 | 128 KB | 240 B |
| kv64 + E2 | 220 | 0 | 64 KB | 0 |
| **kv64 + single-`sp` (E4)** | **182** | **0** | 64 KB | 0 |
| **kv128 + single-`sp` (E4)** | 256 | **4** | 128 KB | 20 B |

### UPDATE (2026-06-12): single-buffering the `sp` union DOES fit kv128

The "no register trick fits kv128" conclusion below was **wrong** — it was based
only on E2 (share `sp_compute`, keep a 2-slot fp8 `p`), which left 126 spills.
Collapsing the **whole `sp` union to a single slot** (E4: `UA_FA4_SINGLE_SP`,
score *and* P single-buffered) drops kv128 spills **173 → 4** (essentially fits).
Attribution: the `sp` score/P double-buffer is the *entire* kv128 register
blocker; `sp_delta` is a red herring (single-buffering it alone = 0 effect, the
compiler already reclaims it). E2 only got to 126 because it kept `p`
double-buffered in a separate array; the true union collapse is far more
effective.

**Correctness:** the single-`sp` build is fully correct as-is. The deferred-PV
`PV(pi)` read and `QK(1-pi)` write now alias the same VGPRs, which is a register
WAR hazard the compiler *must* resolve — it simply serializes PV→QK (the sp
tiles are pure VGPR, no LDS/barrier involvement). Full regression + matrix
(incl. prefill_fp8) PASS with `-DUA_FA4_SINGLE_SP=1 -DUA_PREFILL_D128_BLOCKSIZE=128`.

### But kv128 is a PERF dead-end anyway (measured, not register-bound)

Same shape `b1 hq=hk=5 sq=sk=75600 d128 non-causal`, kernel-only `@perftest`
median, same session:

| config | KV tile | sp | VGPR/spill | LDS | TFLOPs | vs base |
|---|---|---|---|---|---|---|
| baseline | 64 | double | 214/0 | 64 KB | 1632 | — |
| single-`sp` | 64 | single | 182/0 | 64 KB | 1647 | +0.9% (noise) |
| single-`sp` | 128 | single | 256/4 | 128 KB | 1433 | **−12%** |

Two conclusions:
1. **The deferred-PV double buffer earns nothing.** Single-buffering `sp` is
   perf-neutral at kv64 (1632→1647, within noise) while freeing **32 VGPR**.
   The PV‖QK MFMA overlap it exists to enable is not measurable (one MFMA
   pipe/SIMD serializes PV+QK regardless). The 32 VGPR are free to reclaim at kv64.
2. **kv128 fits but runs ~12% slower — it is MEMORY-LATENCY-bound, not
   occupancy- or softmax-bound.** Occupancy is *identical* between kv64 and
   kv128: both run **1 workgroup/CU = 2 waves/SIMD** (8 waves/WG over 4 SIMDs).
   kv64 at 208 VGPR already can't fit a 2nd WG (4 waves×208 = 832 > 512-VGPR/SIMD
   budget), and kv128 at 256 VGPR sits at exactly 512/512 — still 1 WG/CU. The
   earlier "LDS halves occupancy" claim here was WRONG (occupancy was already
   VGPR-capped at 1 WG).

   The ATT overlay (single SIMD, sq=75600, 4 iters; `runs/att_kv64sp` vs
   `runs/att_kv128sp`) shows the regression is entirely in the **memory** phases.
   Per-tile cycles go 1360 (kv64) → 3095 (kv128) = 2.28× for 2× the keys
   (~14%/key). WG1 busy-cycle ratios kv128/kv64:

   | phase | kv64 | kv128 | ratio |
   |---|--:|--:|--:|
   | memwait | 836 | 4244 | **5.1×** |
   | load | 60 | 660 | **11×** |
   | matrix | 476 | 1028 | 2.16× (≈ per-key neutral) |
   | lds | 400 | 924 | 2.31× |
   | softmax | 1044 | 1648 | 1.58× (**amortizes**) |
   | other | 1024 | 1544 | 1.51× |
   | barrier | 888 | 1108 | 1.25× |

   softmax/other/barrier all grow **sub-2×** (they amortize), matrix/lds grow
   ~2× (neutral/key); only `memwait`/`load` blow up super-linearly. Pure
   memory-wait per tile goes 15% → **34%** of the tile. Same total DRAM bytes,
   same occupancy, but each tile's K/V load is 2× larger and the fixed 2-wave/SIMD
   parallelism can't hide the longer latency, so it is exposed as stall.
   MATRIX‖SOFTMAX overlap also drops 10% → 7%.

   This also reconciles the old "kv128 ~10× slower" number: that was
   **spill-driven** (173 spills thrashing scratch). Single-`sp` removes the
   spills, leaving a 12% *memory-latency* loss.

**Corollary for the load-decoupling path (below):** widening only the K/V load to
128 keys gives the *same* doubled per-tile memory latency at the same occupancy,
so it would hit the same memwait wall unless it also raises in-flight load
parallelism (more buffering / higher occupancy). The kernel is already
memory-latency-exposed at kv64 (memwait+load+barrier is a large share of every
tile). The real lever is **latency hiding / occupancy**, not the KV tile width.

## Per-wave register maps (both kernels split M the same: 8 warps x 32 rows)

ASM `mi350_fmha_hd128_fp8.py` explicit slots, kv128, ~224 VGPR / 0 spill:
- S == P **unioned**, single-buffered: 64
- R (O accumulator): 64
- K == V **unioned**: 32
- Q: 16
- softmax + addressing: ~48

CK-UA registers that scale with kBlockN:
- `sp` = `statically_indexed_array<union{sp_compute(fp32), p(fp8)}, 2>`
  (2 slots, each sized at the fp32 score tile). At kv128 the two fp32 score
  tiles are ~128 VGPR even though the deferred-PV pipeline only keeps ONE fp32
  score live at a time.
- `o_acc` = PV C-tile [32 x 128] fp32 = 64 VGPR. **Fixed** (does not scale with kBlockN).
- `kv_tile.k_tile` + `.v_tile` (separate, for K-read/PV overlap).
- mask tile, QK/PV operand staging, and the KV addressing state — all grow with kBlockN.

## What moves the needle (and what doesn't)

- **E2 (share one fp32 `sp_compute` across both slots, keep a 2-slot fp8 `p`
  ping-pong)**: cut kv128 spills 173 -> 126 — the single biggest lever — and is
  **correctness-equivalent to baseline** (a full E2 build reproduces the
  baseline prefill_fp8 output byte-for-byte; same max-abs-delta and identical
  mismatch-element count). It is simply **insufficient**: 126 spills remain.
  At kv64 it is a slight VGPR regression (214 -> 220) since the union already
  let `p` alias `sp_compute`. Left default-OFF as a measured dead end.

  (Process note: an intermediate write-up here claimed E2 *broke* accuracy. That
  was a misattribution — see the pre-existing-failure note below. E2 is correct.)
- **E3 (union k_tile/v_tile, ASM-style)**: 0 effect. The compiler already
  overlaps their live ranges, so unioning is redundant; separating them for the
  K-read/PV overlap costs nothing in VGPR. Keep them separate.
- **Conditional-rescale off / wide-MMA off**: no help (and wide-MMA is a big
  perf win elsewhere, keep on).
- **Paged vs nopage**: paging addressing (tile_scatter_gather, coordinate
  transforms, buffer addressing) costs only ~13-17 spills. Real but not dominant.

## Where the residual ~126 spills live (line-table attribution)

Spills are **diffuse and addressing-heavy**, not one giant tile:
- bit_cast.hpp (fp8 cvt/permute), inlined everywhere — 52
- amd_buffer_addressing_builtins.hpp (K/V DRAM load addressing) — 40
- pipeline.hpp V-load region (~1254) — 28
- tile_scatter_gather.hpp (paged KV) — 26
- tensor_view / coordinate_transform (addressing) — 24 / 23

i.e. once the duplicated score tile is removed (E2), the remaining overflow is
the *genuine doubling of the whole compute working set* (score+P+operands+mask)
plus the grown KV addressing state — there is no single clean cut left.

## Conclusion / recommended path

CK-UA cannot hold a 128-wide **compute** tile under the 256-VGPR ceiling: the
working set at kv64 is 214 and doubling the compute width overflows by ~40-60
live VGPRs spread across many structures. ASM fits kv128 because it
single-buffers the score and hand-packs its register file. The biggest single
CK lever (E2, sharing the fp32 score) only recovers ~47 of the needed spills
and the residual is diffuse, so no combination of register tricks fits kv128 at
128-wide compute. The only path that fits is shrinking the score/operand tiles
themselves: keep the proven kv64 compute footprint (214 VGPR, 0 spill) and
widen ONLY the K/V LDS/DRAM load + block barrier to 128 keys (decouple load
width from compute width), or equivalently sub-tile kBlockN into 2x64 inside
the per-tile phases.

## IMPORTANT: pre-existing prefill_fp8 accuracy failure on this branch

While validating, a CLEAN build of the base branch (jukorhon/fa4-k-preread @
9aa380e6c "wide 32x32x64 FP8 MMA") FAILS the prefill_fp8 correctness check at
the loose fp8 tolerance (atol=rtol=0.15), independent of any kv128/VGPR work:
  - rebuild_and_test matrix `prefill_fp8` (b2 sq8192 h12,2 blk64):
    max abs delta 2.73, 0.6% of elements (140700 / 25,165,824) over tolerance.
  - regression fixtures also report fp8 "not all close" (e.g. 2.2% of elements).
This reproduces byte-for-byte with and without the E2 toggle, so it is NOT
caused by the kv128 experiments. bf16 and fp8 decode paths PASS.

**Bisected to the wide-MMA commit.** A clean build of the PARENT commit
(a4d3ff34f "widen FP8 K/V async loads to dwordx4") PASSES the same prefill_fp8
config; the child 9aa380e6c "wide 32x32x64 FP8 MMA with cvt-only P relayout"
FAILS it. So the 32x32x64 FP8 MMA / cvt-only P relayout introduced a real
prefill_fp8 accuracy regression. This needs separate triage (likely the P
relayout mapping for the wide MMA), and means the "wide MMA closed the gap to
~1.2x" result was measured on a numerically-incorrect prefill_fp8 kernel.

The robust unlock is to **decouple the K/V LDS/DRAM load granularity (128) from
the compute width (64)**: keep the proven kv64 compute footprint (214 VGPR, 0
spill) and only widen the cooperative K/V load + block barrier to 128 keys, so
we capture the DRAM-latency + barrier amortization (the bulk of the kv128 win)
without paying the register cost of a 128-wide compute tile. Full sub-tiling of
kBlockN (compute two 64-halves per loaded 128 tile) is the equivalent inside
the existing per-tile phase machinery. Either keeps all register tiles at their
kv64 size.

Expected ceiling of this path: it captures barrier/DRAM amortization but not the
softmax-overhead amortization (softmax still runs per 64). So the net gain is a
fraction of the naive kv128 promise — weigh against the numeric levers
(exp2 / rowmax-freeze) before investing in the load-decoupling refactor.

## Toggles added (all default OFF, production path bit-identical)

- `UA_PREFILL_D128_BLOCKSIZE` (default 64): compile-time KV tile override for probing.
- `UA_FA4_INPLACE_DELTA` (default 0): drop `sp_delta`, do scale-shift/exp2 in
  place on sp_compute. VGPR-neutral (compiler already reclaims sp_delta), kept
  because SHARED_SPCOMPUTE needs it.
- `UA_FA4_SHARED_SPCOMPUTE` (default 0): one shared fp32 sp_compute + 2-slot fp8
  p. The E2 lever above.
- `UA_FA4_UNION_KV` (default 0): union k_tile/v_tile. VGPR-neutral; kept as a
  documented dead-end.

Probe with: `XCFLAGS="-DUA_PREFILL_D128_BLOCKSIZE=128 -DUA_FA4_SHARED_SPCOMPUTE=1" ua-test-scripts/measure_vgpr.sh`

---

# Softmax-phase latency investigation (canonical fp8 prefill, b1 sq=sk=75600 hq=hk=5 d128 non-causal)

Current production config (committed): kv128 tile + cooperative K/V load + single-sp +
wide 32x32x64 MMA + packed shift + packed alu1 rescale. Standalone kernel-only median
**1782 TF/s** (== ~1774 TF/s on the contiguous prod path; the python paged path is
~1483 TF/s — paging addressing overhead, separate axis). The wide-MMA fp8 accuracy
regression flagged below is **FIXED** (CK f947db93f "fix wide-MMA FP8 P relayout":
the cvt-only P relayout was missing the QK-C->PV-A transpose); full regression matrix
+ standalone check now PASS, and the python harness reports CK-vs-ref PASS.

## What landed (committed, CK 29e0f75e1)
- **Packed score shift** (`UA_FA4_PACKED_SHIFT`): 64 scalar `v_fma_f32` -> 32
  `v_pk_fma_f32` (addend `-scale_s*max` is per-thread uniform). +4.5%, bit-identical.
- **Packed alu1 o_acc rescale** (`UA_FA4_PACKED_ALU1_RESCALE`): 6 `v_mul_f32` ->
  3 `v_pk_mul_f32`. +4%, bit-identical. Net **+8%** over the prior 1649 baseline.

## Softmax COMPUTE is already hidden — it is not the wall-time gate at this shape
Per-warp-group ATT phase breakdown (busy+stall, sq=8192 trace, repeated per tile):

| phase | exec | stall | %wave |
|---|--:|--:|--:|
| softmax | 72747 | 39628 | **33.8%** |
| barrier | 1161 | **52720** | 16.2% |
| memwait | 6426 | **41508** | 14.4% |
| matrix | 7041 | 28732 | 10.8% |
| lds | 18760 | 12300 | 9.4% |

softmax has the largest *exec*, but matrix is only 10.8% (the user's "matrix is hidden
under softmax" is correct) — and the largest *stalls* are barrier (52720) + memwait
(41508) = the ping-pong rendezvous waiting on K/V DRAM latency. softmax exec overlaps
the peer warp-group's matrix/load, so cutting softmax *instruction count* mostly
converts to more barrier stall rather than less wall time. This matches the kv128
memory-latency-bound conclusion above.

Evidence (all at the real sq=75600 shape, kernel-only median, 3+ runs each):
- **exp2 Schraudolph approx** (`UA_FA4_EXP2_APPROX=1`, replaces 64 quarter-rate
  `v_exp_f32` with full-rate `v_cvt_u32_f32`): **-11%** (1782 -> 1587). The cvt/finish
  path adds ALU on the critical chain and the exp throughput it removes was hidden.
  Hard loser; default OFF.
- **Split rowmax ahead of PV gemm** (`UA_FA4_SPLIT_ROWMAX=1`): emit `fmha_alu0_rowmax`
  before `gemm_1(PV)` and `fmha_alu0_shift` after, so the MFMA cluster hides the
  reduce->shift-addend chain. **Neutral** (1782 either way), bit-identical. The post-RA
  scheduler already groups instructions per the core-loop `sched_group_barrier` hints,
  so source-level reorder does not move the emitted schedule. Kept as a default-OFF
  structural hook (`fmha_alu0` is split into rowmax/shift lambdas; the combined call
  is byte-for-byte the original). Moving the rowmax earlier in the *schedule* would
  require editing the `sched_group_barrier` hints, not the call order.

## Recommended next lever
softmax compute is saturated against the overlap; the real headroom is the
barrier+memwait stall (~30% of the wave), i.e. **memory-latency hiding** — deeper K/V
prefetch / more in-flight loads / higher occupancy — consistent with the kv128
"memory-latency-bound, not softmax-bound" finding above. The numeric softmax shortcut
(rowmax-freeze / conditional rescale, already on via `CONDITIONAL_RESCALE=1`) is the
only softmax-side lever that removes critical-path *dependency* rather than hidden exec.
