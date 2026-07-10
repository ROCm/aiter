# gfx950 Gluon unified-attention: MFMA size + decode tuning

Method (all numbers): torch.profiler, per-kernel-name filtering, **512 MB L2/Infinity-cache
flush between every iteration**, 30 iters. bf16, HEAD_SIZE=128, causal. Configs come from
the Triton heuristics (`select_2d_config` / `select_3d_config`); decode = split-KV attention
+ Triton `reduce_segments` (identical reduce on both sides). Triton = reference.

## 1. 16×16 vs 32×32 MFMA  → **32×32 for prefill, 16×16 for decode**

| workload | 32×32 | 16×16 | verdict |
|---|--:|--:|---|
| prefill b1 8192/8192 64/8 | **851 TFLOP/s** | 401 (BM64) / 389 (BM128) | **32×32 ~2× faster** |
| prefill b8 1024/1024 8/1  | **359 TFLOP/s** | 234 | 32×32 faster |
| decode GQA (large)        | 1.00× | 1.00× | tie (memory-bound) |
| decode MQA (8/1)          | 0.78–0.93× | **0.82–0.99×** | **16×16 +4–6 pts** |
| decode small (C16 ctx1024)| 0.80× | **0.85×** | 16×16 better |

- **Prefill is compute-bound** → 32×32's higher MFMA throughput wins decisively (16×16 needs
  ~2× the instructions for the same tile).
- **Decode**: `BLOCK_M=16` (16×16) matches Triton, halves GQA/MQA row-waste, fewer VGPRs/spills.
  Neutral for large GQA (memory-bound, waste hidden) but a clear win for MQA and small batches,
  where per-row compute/overhead is exposed. Never worse than 32×32 on any decode shape.
- ⇒ **prefill = 32×32 / BLOCK_M=128; decode = 16×16 / BLOCK_M=16 / num_warps=1** (both wired into
  the host wrapper via `max_seqlen_q==1`). `MFMA_DIM` is a kernel knob (default 32). Full 16-vs-32
  decode sweep in `decode_mfma16.md`.

## 2. Decode tricks from gfx1250 (ALL_DECODE fast-path + block-table clamp)

- one-program-per-sequence fast path (no binary search, grid dim1 = num_seqs), block-table
  read clamped (decode-only, so prefill inner loop is byte-identical).
- Impact: **1.01–1.04×** over no-tricks — small (decode is memory-bound; these cut launch/index
  overhead, not memory traffic).

## 3. Why gluon decode trails Triton — TTGIR + register diff (C64 ctx8192 64/8)

| metric | Triton 3d | Gluon 32×32/BM32 | Gluon **16×16/BM16** (current) |
|---|--:|--:|--:|
| VGPR / spills | 180 / 0 | 256 / **7** | **236 / 0** |
| LDS | **32 KB** | 64 KB | 64 KB |
| num_warps | 2 | 1 | 1 |
| occupancy (64 KB/CU) | **2 wg/CU** | 1 | 1 |
| MFMA enc | mfma v4 [16,16,32] | mfma v4 [32,32,16] | mfma v4 [16,16,32] (= Triton) |
| KV→LDS | `tt.load`→`local_store`, **single-buf** | async, double-buf | async, double-buf |

Two effects: (a) **16×16 removed the 7 register spills** (256→236 VGPR, 0 spills) — part of the
MQA/small-batch win. (b) The **remaining gap is LDS**: gluon explicitly double-buffers full K+V
(`NUM_BUFFERS=2`, direct-to-LDS async) → **64 KB → 1 wg/CU**, while Triton single-buffers KV (32 KB)
and pipelines via `num_stages=2` → **2 wg/CU**. That 2× occupancy is why Triton still edges ahead
where workgroups are scarce (MQA / small batch); large GQA has enough workgroups to saturate BW at
occupancy 1 (→ parity). Raw IR: `ir_triton3d_dec.ttgir`, `ir_gluon_dec16.ttgir`.

**Confirmation** — forcing 32 KB LDS via `TILE_SIZE=32`:

| shape | Triton | gluon TILE64 (64 KB) | gluon TILE32 (32 KB) |
|---|--:|--:|--:|
| C16 ctx1024 64/8  | 3169 GB/s | 2483 (0.78×) | 2405 (0.76×) |
| C64 ctx8192 64/8  | 5772 GB/s | 5627 (0.97×) | 5552 (0.96×) |
| C128 ctx8192 64/8 | 5755 GB/s | 5314 (0.92×) | **5600 (0.97×)** |

Lower LDS recovers the large-batch gap (C128 0.92→0.97×) but the 2× tile count hurts small
shapes — so it's a tradeoff, not a free win. A single-buffered decode LDS scheme (32 KB at
TILE=64) would close it without the tile-count penalty, but is a structural change.

## 4. Decode: gluon vs Triton — FINAL (matched grid, 32×32, tricks on). Full 24-shape sweep in `decode_final.md`.

**GQA (64/8): at/above parity for ctx ≥ 4096** (the earlier gap was purely grid order):

| shape | Triton GB/s | Gluon GB/s | speedup |
|---|--:|--:|--:|
| C64  ctx8192 64/8 | 5734 | 5789 | **1.01×** |
| C128 ctx8192 64/8 | 5661 | 5666 | **1.00×** |
| C32  ctx8192 64/8 | 5329 | 5380 | **1.01×** |
| C64  ctx4096 64/8 | 5342 | 5303 | 0.99× |
| C16  ctx1024 64/8 | 3213 | 2540 | 0.79× (tiny, overhead-bound) |

**MQA (8/1): still 0.78–0.98×** — grid order can't help (1 kv_head ⇒ seq-fast == kv_head-fast),
so the MFMA-forced `BLOCK_M=32` (vs Triton 16) row-waste + 64 KB LDS dominate:

| shape | Triton GB/s | Gluon GB/s | speedup |
|---|--:|--:|--:|
| C64 ctx8192 8/1 | 4669 | 4283 | 0.92× |
| C32 ctx8192 8/1 | 3796 | 2948 | 0.78× |

## 5. Prefill — no regression

Gluon b1 8192/8192 = **849 TFLOP/s** (pre-tuning 851). The block-table clamp is gated to
`ALL_DECODE`, so prefill's inner loop is unchanged; MFMA_DIM defaults to 32.

## 6. Grid order — the decode configs were NOT identical (now hardcoded via `ALL_DECODE`)

Triton 3d launches `(num_seqs, num_kv_heads, num_segments)` (seq fastest-varying); gluon launched
`(num_kv_heads, q_blocks, splits)` (kv_head fastest). The optimal order is fully determined by
`ALL_DECODE` (decode → seq-fastest, prefill → kv_head-fastest, which already matches Triton 2d), so
the kernel now derives the program-id mapping from `ALL_DECODE` and the host launches the matching
grid order — no separate knob. Prefill (2d grid) is a separate launch and is untouched. Matching the
decode order recovers the large-GQA gap:

| shape | Triton | gluon kv_head-fast | gluon **seq-fast (matched)** |
|---|--:|--:|--:|
| C64 ctx8192 64/8  | 5763 GB/s | 5645 (0.98×) | **5812 (1.01×)** |
| C128 ctx8192 64/8 | 5755 GB/s | 5386 (0.94×) | **5715 (0.99×)** |
| C16 ctx1024 64/8  | 3179 GB/s | 2501 (0.79×) | 2553 (0.80×) |
| C64 ctx8192 8/1   | 4671 GB/s | 4324 (0.93×) | 4344 (0.93×) |

seq-fastest ⇒ parity for large GQA; no effect on MQA (Hkv=1, head dim trivial) or tiny shapes.

### Remaining decode config differences (all MFMA-forced, not tunable without a 16×16 path)
| knob | Triton | Gluon | note |
|---|--:|--:|---|
| BLOCK_M | 16 | 32 | gluon 32×32 MFMA ⇒ BLOCK_M = 32·num_warps |
| num_warps | 2 | 1 | follows BLOCK_M |
| TILE_SIZE / num_seg / reduce | 64 / S / same | 64 / S / same | shared |

## 7. num_warps sweep (decode) — `num_warps=1` is optimal

Tried matching Triton's `num_warps=2`. With 16×16 MFMA, `BLOCK_M = 16·num_warps`, so nw2⇒BM32,
nw4⇒BM64. Result: nw1 is best or tied everywhere; nw2 ~neutral (occasionally +1 pt MQA); nw4 worse.

| shape | Triton nw2/BM16 | gluon nw1/BM16 | nw2/BM32 | nw4/BM64 |
|---|--:|--:|--:|--:|
| C64 ctx8192 8/1 | 4595 | **4410 (0.96×)** | 4387 (0.95×) | 3961 (0.86×) |
| C16 ctx8192 8/1 | 3078 | **2670 (0.87×)** | 2654 (0.86×) | 2357 (0.77×) |
| C64 ctx8192 64/8 | 5720 | **5786 (1.01×)** | 5613 (0.98×) | 5449 (0.95×) |

VGPR: nw1 236 / nw2 188 / nw4 172 — but **LDS = 64 KB for all** (independent of num_warps).
- Triton decouples num_warps from BLOCK_M (2 warps on a 16-row tile, no waste). Gluon's MFMA
  layout couples them, so nw≥2 just adds wasted decode rows → cancels the extra-wave benefit.
- num_warps can't raise occupancy (LDS-bound at 1 wg/CU regardless). ⇒ keep **num_warps=1**;
  the only remaining lever is 64→32 KB LDS (single-buffered decode K/V).

## 8. Single-buffered 32 KB decode LDS → occupancy 2 (implemented, decode default)

Prototype confirmed the §3 diagnosis. `attention_loop_single_buffer` keeps one K + one V tile
in LDS (32 KB, VGPR 236→200, no spills) with a `thread_barrier` between tiles; occupancy 1→2.
No in-workgroup overlap, but the extra resident WG provides cross-WG overlap. Decode-only
(guarded: only masks the last tile). Wired as the decode default (`num_buffers=1` when
`max_seqlen_q==1`); prefill stays double-buffered.

| shape | Triton | double-buf 64 KB | **single-buf 32 KB** |
|---|--:|--:|--:|
| C32 ctx8192 8/1 | 3796 | 3153 (0.83×) | **3489–3545 (0.92–0.95×)** |
| C16 ctx8192 8/1 | 3077 | 2673 (0.87×) | **2888 (0.94×)** |
| C16 ctx1024 64/8 | 3250 | 2725 (0.84×) | **3003 (0.92×)** |
| C64 ctx8192 64/8 | 5768 | 5787 (1.00×) | **5845 (1.01×)** |
| C128 ctx8192 64/8 | 5766 | 5699 (0.99×) | **5758 (1.00×)** |

MQA/small-batch +7–9 pts, large GQA parity-or-better, no regressions. Eval: `decode_singlebuf.md`.

## 9. Triton 3.6.0 vs 3.7.1 (same kernel, forced BlockedLayout path, decode nb=2)

Two stock-3.7.1 portability issues found first: (a) 3.7.1's `async_copy` (buffer *and* global)
requires `BlockedLayout`/`SliceLayout` offsets, so the kernel's `TRITON_BEYOND_37=True` path
(`DistributedLinearLayout` from `_offset_bases_to_blocked`) **does not compile on stock 3.7.1**;
(b) `gl.thread_barrier` **does not exist in 3.7.1** (single-buffer decode won't compile there).
To compare the compiler versions on identical logic, forced the BlockedLayout path + nb=2 decode.

| workload | triton 3.6.0 | triton 3.7.1 |
|---|--:|--:|
| PREFILL b1 8k 64/8 (gluon) | **851 TFLOP/s (1.16×)** | 826 TFLOP/s (1.10×) |
| DECODE C64 ctx8192 64/8 | 5749 GB/s (1.00×) | 5795 (1.01×) |
| DECODE C128 ctx8192 64/8 | 5696 (0.99×) | 5713 (0.99×) |
| DECODE C32 ctx8192 8/1 | 3193 (0.84×) | 3266 (0.85×) |
| DECODE C16 ctx8192 8/1 | 2667 (0.86×) | 2713 (0.87×) |

Verdict: **no material difference** — 3.7.1 prefill is ~3% slower for gluon, decode is ~1–2% faster;
gluon-vs-triton ratios are essentially unchanged. Env restored to 3.6.0 (3.6.0 wheel + `pip_freeze.txt`
saved in `snapshots/`). Note 3.7.1 was only usable via the forced-Blocked path; shipping the modern
`>=3.7` path on stock 3.7.1 needs `_offset_bases_to_blocked` to emit a `BlockedLayout` and a
`thread_barrier` replacement.

## 10. store_partial buffer_store + Q-through-LDS (both tested, then reverted)

Neither improved perf, so both were removed to keep the kernel clean.

- **buffer_store in `store_partial`**: perf-neutral (±1–2%, within noise — the partial write is
  tiny vs KV reads). Reverted to plain `gl.store`; callers pad the partial buffers' token dim so
  masked decode lanes stay in range (as the benchmark's `alloc_segm` does).
- **Q-through-LDS** (coalesced `blocked_q` load + `convert_layout` LDS reshuffle vs direct dot-layout
  load): does not help — prefill 1.00× (neutral), decode neutral-to-worse (C16 8/1 0.93→0.85×;
  decode Q is one token, so the LDS round-trip costs more than the coalescing saves). Removed.

## 11. Split count for MQA decode — Triton heuristic OVER-splits for gluon

Sweeping `NUM_SPLITS` (8/1, gluon nb1/nw1). 512 resident WG slots (256 CU × 2 wg/CU).

| shape | Triton@heur | gluon@heur | gluon best | opt WGs |
|---|--:|--:|--:|--:|
| C16 ctx8192 8/1 | 3002 (S128) | 0.96× | **S32 1.08×** | 512 |
| C32 ctx8192 8/1 | 3814 (S128) | 0.91× | **S32 1.15×** | 1024 |
| C64 ctx8192 8/1 | 4631 (S64) | 0.94× | **S16 1.14×** | 1024 |
| C128 ctx8192 8/1 | 5068 (S32) | 0.99× | **S16 1.08×** | 2048 |

- Underoccupancy is real at the low end (C16 S16 = 256 WG < 512 slots → 0.92×; S32 fills it → 1.08×).
- But the Triton heuristic (sized for nw=2 occupancy) overshoots to 2048–4096 WG (4–8× slots); past
  ~1× fill, extra splits only add overhead (Q reload per split, fewer tiles/split, +1 reduce segment).
- Sweet spot ≈ **1–2× resident slots (~512–1024 WG)** → gluon **1.08–1.15× faster than Triton**.
- Fix (dispatch-side, gluon-specific): `NUM_SPLITS = clamp(round(TARGET_WGS/(num_seqs*num_kv_heads)),
  1, num_tiles)`, `TARGET_WGS ≈ CU*2..4`. MQA-specific — GQA's larger per-split KV already makes the
  heuristic split count fine (GQA was at parity).

**Small ctx (1024, 8/1)** — splits capped at num_tiles=16, KV fits in cache (overhead/latency-bound):

| shape | Triton us | gluon best |
|---|--:|--:|
| C16 ctx1024 8/1 | 9.8 | 0.98× (S16, capped — underoccupied at 256 WG) |
| C64 ctx1024 8/1 | 14.1 | 0.93× (S16) |
| C128 ctx1024 8/1 | 22.7 | **S4 1.04×** |
| C256 ctx1024 8/1 | 40.0 | **S4 1.18×** |

Right-sizing wins for large batch (C128/C256 → 1.04–1.18×) but **small batch (C≤64) loses (0.93–0.98×)**:
num_tiles=16 caps splits below the 512-slot fill point (C16 max 256 WG → structurally underoccupied),
and the tiny cached KV makes it overhead-bound where Triton's nw=2 + lower fixed cost wins. Only lever
for that corner is a smaller TILE_SIZE (more tiles → more splittable WGs), but upside is small.

## 12. page_size 32 vs 64 (gluon TILE=page; page=32 -> 16KB LDS, occ 4, 2x num_tiles)

gluon @ right-sized splits, triton @ heuristic, per page (decode GB/s):

| shape | gluon p64 | gluon p32 | triton p64 | triton p32 |
|---|--:|--:|--:|--:|
| C16 ctx1024 8/1 | 836 | 797 (−5%) | 840 | 866 |
| C128 ctx1024 8/1 | 3289 | 3282 | 2999 | 2569 (−14%) |
| C16 ctx8192 8/1 | 3106 | 3301 (+6%) | 3078 | 2836 |
| C128 ctx8192 8/1 | 6007 | 5765 (−4%) | 5018 | 4802 |
| C64 ctx8192 64/8 | 6165 | 6210 | 5648 | 5709 |

- gluon page=32 ≈ **neutral** (±few %); only C16 ctx8192 (+6%) benefits — the one occupancy-starved
  *and* BW-bound case. LDS drops to 16 KB (occ 4) but occupancy isn't the binding limit elsewhere.
- **Does NOT fix small-ctx small-batch** (C16 ctx1024 −5%): overhead/cache-bound tiny kernel, more
  tiles just add reduce/launch overhead — no bandwidth to hide, so extra WGs don't help.
- Triton **degrades at page=32** (−4..−16%, worst at small ctx); it prefers larger tiles. So the
  gluon/triton ratio looks better at page=32 mostly because triton regresses.
- Verdict: keep **TILE=page=64** default. page_size is a deployment property anyway; the useful
  finding is gluon is **more robust to small pages** than triton.

## Recommendation

- **Prefill**: 32×32 MFMA / BLOCK_M=128 / double-buffered → **1.0–1.19× faster than Triton**.
- **Decode**: 16×16 MFMA / BLOCK_M=16 / num_warps=1 / **single-buffered 32 KB** / seq-fastest grid
  / split-KV + Triton reduce → **parity for GQA (1.00–1.01×), 0.92–0.95× for MQA/small batch**.
- Grid order auto-derived from `ALL_DECODE` (no knob). All levers (MFMA size, grid order,
  num_warps, decode tricks, LDS buffering) are now tested and set to their best values.
