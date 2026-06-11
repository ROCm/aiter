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
caused by the kv128 experiments. It looks like a real accuracy regression in the
wide-MMA fp8 path (the cvt-only P relayout / 32x32x64 MMA) that predates this
work and should be triaged separately. bf16 and fp8 decode paths PASS.

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
