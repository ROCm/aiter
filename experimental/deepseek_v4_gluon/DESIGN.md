# DeepSeek V4.1-Flash / V4-Pro Gluon monokernels (gfx950)

Gluon ports of vLLM's FlyDSL mono decode layers
(`vllm/models/deepseek_v41/amd/mono`, `vllm/models/deepseek_v4/amd/mono`, vLLM main
`8321d11`), built on what the K3 and MiniMax-M3 work established. **Scope matches
FlyDSL exactly**: where upstream fuses (K1, K2, the FFN launch) we fuse the same
stages; where it splits (vLLM's compressor / indexer between K1 and K2, the FFN
launch for layers whose attention vLLM runs, the draft layers' own runner) we split
at the same places.

## Contract: drop-in for the FlyDSL builders

Each Gluon builder (`build_mono_k1`, `build_mono_k2`, `build_mono_ffn`) returns a
`launch(...)` with **the FlyDSL launch's argument list** (raw int64 addresses, the
rank, the epoch buffer, the timeline record) and uses **the same scratch layout,
peer layout, tags and epoch protocol** (`layer.scratch_layout`,
`peer_half_bytes`, `Mailbox((ep << 1) + 1 | 2)`, `epoch_begin` / `epoch_end`). So:

- upstream's host side is reused unchanged: `DSV41MonoLayer` / runner, scratch and
  peer allocation, `mono_decode.py` docking, graph capture, widths, refusals;
- a Gluon and a FlyDSL launch of the same layer are interchangeable (same tensors
  at the layer boundary), so they interleave per layer and A/B per layer;
- validation is mechanical: run both on identical inputs and diff every mailbox
  region of the scratch (bitwise where the arithmetic order is ported, else
  numerically), then the layer outputs, then vLLM's native modules.

Host-side layout math is imported from upstream, never re-derived (a copied
layout that drifts is the bug class "region sized for 8 rows overflowed at 16").

## Milestones (V4.1-Flash first; V4-Pro is V4.1 plus deltas)

| # | Deliverable | Validation |
|---|---|---|
| M0 | Gluon framework: tagged put / batched poll (device sc1, system sc0 sc1), bounded waits + sticky health word, publish (drain + barrier + put), epoch begin / end + marks, peer bases, timeline stamps, compile probe (VGPR / spills / LDS / occupancy) | unit kernels on 1 and 4 GPUs: tag round trips, cross-XCD, cross-rank, timeout path |
| M1 | V4.1 **FFN launch**: push, FFN seam (reduce + slice, gate + Sinkhorn, norm), MoE (xq, router, route, shared, ug, down, reduce), epoch end | per-region diff vs FlyDSL `build_mono_ffn`; layer out vs vLLM `DeepseekV4MoE` + seam; TP2 / TP4, S = 1..48 |
| M2 | V4.1 **K2** = attention back (split attention over the key table, combine + inverse RoPE, wo_a, wo_b, push) + M1's FFN half | per-region diff vs FlyDSL K2 |
| M3 | V4.1 **K1** = attention seam (slice, gate, norm -> vLLM MXFP8) + attention front (wqkv, q / kv norms + KV insert, wq_b + RoPE); index layers hand normed input + qr to vLLM | per-region diff vs FlyDSL K1; KV records byte-equal |
| M4 | V4.1 DSpark draft layers (128 experts / top-3, one FP4 group per ug task, SWA-only attention) | as M1-M3 on the draft runner |
| M5 | V4-Pro deltas: hidden 7168, TP8, S <= 64, 2.5 routed scale, hash layers 0-2, CSA / HCA key tables (`global_keys`, `bounded`), two-hop all-reduce from 42 rows | as M1-M3 vs `deepseek_v4/amd/mono` |
| M6 | Docking: `--monokernel` with a backend switch (FlyDSL / Gluon) per launch kind; e2e GSM8K / GPQA + AgentX vs FlyDSL | engine tests of `tests/models/test_dsv41_mono_*` with the Gluon backend |

### Status (2026-10-28)

| # | State |
|---|---|
| M1 FFN launch | done; bitwise = FlyDSL at TP4 and TP2, S = 1, 4, 16, 17, 48 (all scratch regions + outputs, plain regions poisoned) |
| M2 K2 | done (non-indexed); bitwise = FlyDSL at TP4 and TP2, ratio 1 (compressed + window keys); indexed build (`stage_kt` + KRDY) compiles, untested |
| M3 K1 | done (non-index source); bitwise = FlyDSL at TP4 and TP2, S = 1, 4, 16, 17, 48: seam outputs, LIN / PMIX / QKVP, X8 / QX8, q, key table, KV cache records; index source (normed / qr out) not yet |
| M4-M6 | not started |

Perf (CUDA graph, TP4, 20 launches, median; same inputs every launch): FFN
Gluon / FlyDSL 1.46 (S = 1) .. 1.80 (S = 48); K2 1.60 .. 1.88; K1 1.29 .. 1.68
(12.5 vs 18.7 us at S = 1). Correctness-first
port: no weight-before-poll issue order, no LDS padding, write-through (sc0
sc1) where upstream stores device scope.

## Learnings applied (from K3 Gluon, M3 FlyDSL / Gluon, the DS mono)

Kernel design
- 256 CTAs x 512 threads, one CTA a CU, all resident; refuse a width whose
  compiled occupancy cannot hold the grid (`hipModuleOccupancy...`, as M3's
  `residency.py`) or that spills (spills are a correctness risk in a spinning grid).
- Stage walk, not a task table: every CTA walks every stage in program order over
  its `first_task(bid, base)` tasks; consumers only wait on earlier stages
  (deadlock-free while resident). Phases inline; per-width specializations
  (constexpr S) keep each width's register budget honest.
- Placement is the schedule: stages that can start early sit on CTAs the critical
  path does not use (router behind xq, shared expert behind route, ug round 0
  after the shared units), exactly upstream's bases (`rs.router0`, `route0`,
  `shared0`, `ug0`).

Synchronization
- Tagged pairs (value | tag << 32, one 8 B store) for every small hand-off;
  plain words + drain + barrier + one flag for bulk (MXFP8 rows, MID).
- Tag = launch-pair epoch (K1 `2e + 1`, K2 / FFN `2e + 2`); nothing is zeroed
  between steps; peer halves alternate by epoch parity; the epoch moves on in
  K2's / FFN's CTA 0 after every CTA's mark (graph-safe, no host step hook).
- Polls are side-effecting wave-local asm (`global_load_dwordx2 ... sc1`, sc0 sc1
  for peer memory) with a memory clobber, batched (<= 12 pairs a thread a round
  trip); payload reads behind a flag carry an opaque dependency on the poll so
  LLVM cannot hoist them above it.
- Bounded waits (100 MHz `s_memrealtime`, 10 s) with a sticky health word: a
  timeout ends every later wait at once, so a stall becomes a report (M3's
  fail-stop), not a hang.
- A CTA barrier after a flag wait that guards other waves' loads; publish =
  `s_waitcnt vmcnt(0)` + barrier + put (a workgroup release does not drain
  vmcnt outside tgsplit).

Scheduling ops
- Issue weight loads before polling activations (`*_loads` then poll then MFMA),
  publish before prefetch, keep polling CTAs free of bulk loads in flight.
- Size tasks to the grid at the realistic expert-union size; ug queue past one
  token tile (`take_ug_task`), down's extra task to the CTAs that finish first
  (`take_down_rank`).
- Numerics are the reference's: rank-ordered fp32 sums, aiter stage-2 rounding
  (bf16 at every add, top-k order, shared last), mHC fold order, Sinkhorn with
  hw rcp / exp2, vLLM MXFP8 codes.

Gluon specifics (Triton 3.8 in the vLLM nightly image)
- No `gl.atomic_load` / `atomic_store`: polls and tagged puts are inline asm;
  `buffer_load(..., cache=".cv")` for coherent int payloads (`gl.load .cv`
  scalarizes).
- `s_nop` wait states before any asm that reads an MFMA result.
- Addresses are pointer parameters (`*u8`, normalized with `u8()` at entry;
  a raw address goes in through `ffn.ptr`), never ints cast in-kernel: Triton
  knows a pointer argument's 16-byte alignment and vectorizes off it, while an
  `int -> pointer` cast loses it -- byte loads, and past S = 4 the spills that
  came with them (38 at S = 6..16, 0 after). `ptr.to(gl.uint64)` gives the
  integer the asm polls / puts take.
- Load each MFMA operand in the MFMA's own lane order, so the conversion to the
  dot-operand layout is free: A `[wave][row][k]` with lane = row + 16 kq
  (`BlockedLayout([1,1,16],[1,16,4],[8,1,1],[1,2,0])`), B `[wave][k][token]`
  with lane = token + 16 kq (order `[2,1,0]`), scales a byte a lane, lane = row
  + 16 group. The other thread order costs 8 `ds_bpermute` a 32-row tile, and a
  scale tensor in another order an LDS round trip, at every MFMA.
- aiter's shuffled E8M0 scales: load them in their stored order
  (`[wave][col e][row b][half a]`, the two 16-row halves' codes adjacent: one
  2-byte load a lane), then permute + reshape to `[wave][32][4]`: that is the
  scale operand's layout exactly (no conversion).
- A mask, not an index clamp, for ragged K: `min(k, K - 1)` is not contiguous
  (byte loads), `mask=k < K` keeps 16-byte loads.
- In a `static_range` body a constexpr cannot be re-bound, and `min` / `max`
  in jit code are `gl.minimum` / `maximum` (tensors): a
  `gluon.constexpr_function` for constexpr arithmetic.
- Cross-CTA data must be stored write-through: MI355X has 8 XCDs, each its own
  L2, and a plain store stays in its XCD's L2 -- a consumer on another XCD reads
  memory and sees the old bytes even after the producer's flag. Published bulk
  data: `gl.store(..., cache_modifier=".wt")` (sc0 sc1, vectorized) or the
  sc1 asm stores; consumers read plain (the lines are new to their L2). Found
  by poisoning the plain regions between the reference and Gluon launches.
- Per-call ints `do_not_specialize`; warm-up with exact-dtype stand-ins.

## Layout of this directory

```
common/        Gluon framework: sync.py (tagged put / bounded batched polls, publish,
               health), epoch.py, ops.py, mx.py, probe.py (GPU-free compile: VGPR /
               spills / LDS / ISA)
v41/           V4.1-Flash: seam.py (FFN / attention seam), moe_front.py (xq, router,
               route), moe_shared.py, moe_ug.py, moe_down.py, plan.py (route shape),
               ffn.py (the FFN launch, ffn_half), back.py + k2.py (K2), front.py +
               k1.py (K1)
tests/         probe_*.py: CPU compile probes; gpu_{ffn,k1,k2}_vs_flydsl.py: the
               bitwise comparisons against FlyDSL on its own runner
triton_patch/  the Triton fix the port needs (rank-3 mfma_scaled) + build.sh / Dockerfile
```

## Running

1. Image: `triton_patch/build.sh` (ROCm/triton 669b31ac + `rank3_mfma_scaled.patch`,
   built as a wheel in the vLLM nightly 00b6c4c5 image, installed into an image
   FROM it; ~25 min).
2. Compile probes, no GPU: `./cpu_run.sh -m deepseek_v4_gluon.tests.probe_m1_loads 1 16 48`
   (also `probe_m2_k2`, `probe_m3_k1`).
3. GPU comparisons: `CARDS=0,1,2,3 ./gpu_run.sh torchrun --nproc-per-node=4 -m
   deepseek_v4_gluon.tests.gpu_k2_vs_flydsl 1 4 16 17 48` (`BENCH=1`: CUDA-graph
   timing). Needs `UPSTREAM` (vLLM 8321d11's `deepseek_v41/amd/mono`, mounted over
   the image's older copy) and the checkpoint config at /data/DeepSeek-V4.1-Flash.
   `gpu_when_free.sh` waits for 4 idle GPUs first (shared boxes).

## GEMVs: batched scaled MFMA (Triton patch)

Every MXFP4 / MXFP8 GEMV of the reference gives each wave independent work (its
own K slice, or its own expert). Gluon expresses that as a rank-3 (batched) MFMA,
`[waves, M, K] x [waves, K, N]` with `warps_per_cta=[8, 1, 1]`. Triton 3.8
(ROCm/triton 669b31ac, the vLLM nightly's) supports it for plain `mfma` but
aborts for `mfma_scaled`: `chooseScaledMfmaScaleLayout` builds a rank-2 scale
layout whatever the rank (the lowering, `convertScaledDot`, is batch-aware).
The fix (`~/triton-mfma-scaled`, 24 lines): K from the last dim, size-1 batch
bases on the tile, batch the slowest warp dim, a rank-matched CGA layout; rank 2
unchanged. GPU test: `test_amd_mfma_scaled_batched` in `python/test/gluon/test_core.py`.

Decomposition per GEMV:
- down: a wave a (task, slot of U): both operands from global (w2 rows, the
  picks' MID rows) at per-wave addresses -- FlyDSL's split, one to one.
- shared expert (and router-like GEMVs): wave w takes K steps w, w + 8, ..; the
  activation tile in LDS is read through a 3-D view [waves, K window, tokens]
  whose batch stride is the step -- FlyDSL's split, one to one.
- ug: FlyDSL splits K in two halves per (group, half) pair (ks = wave / 2 % 2),
  not linear in the wave, so no LDS view expresses it. Here K splits 8 ways and
  every wave covers the task's 128 rows: the same 40 MFMAs a wave, each x byte
  read from LDS once (80 KB a task, FlyDSL 320 KB), a larger cross-wave
  reduction, and a different fp32 summation order (validated numerically).

## K2 attention back (M2): decomposition

`v41/back.py` + `v41/k2.py` (K2 = back, then `ffn.ffn_half(PUSH=False)`: wo_b
pushes its bf16 pairs into every rank's ATTN region exactly as the FFN launch's
`stage_push` does, so the seam is unchanged).

- split: the QK is a batched bf16 MFMA `[groups][keys][32] x [groups][32][16
  heads]` (warps `[NG, 8 / NG, 1]`: a wave a 16-key tile); each 32-dim chunk's K
  positions are upstream's (`64 c + 16 g + 8 h + i`), q is kept in LDS
  chunk-permuted so a chunk is one `.index()`. Scores' max is exact in any
  order; the 16-key sums follow upstream's tree (reshape keys to `[2,2,2,2]`
  and sum the 2-wide axes in its order), the tiles' sums fold in tile order
  through LDS. PV is a 2-D MFMA `[128 = 16 w + r][16 keys] x [16 keys][16
  heads]`, one per dt plane (dims `64 w + 4 r + dt`), the raw records gathered
  from LDS (wave 7: the bf16 rope dims).
- combine: `[8 heads][64 lanes][8 dims]` tensors, upstream's order (unfused
  mul / add, IEEE division -- Triton lowers `/` to div_scale / fmas / fixup).
- wo_a / wo_b: the batched scaled-MFMA GEMV with K split as upstream: wo_a
  four contiguous K parts x two 16-row tiles (warps `[4, 2, 1]`), wo_b steps
  interleaved over the 8 waves; wave-ordered sums from 0.0.

Validation (`tests/gpu_k2_vs_flydsl.py`, `tests/gpu_ffn_vs_flydsl.py`): FlyDSL
first, then Gluon on the same buffers; every plain-data region is poisoned
(0xFF) in between, so a region Gluon fails to write cannot pass as FlyDSL's
leftover (a tagged region it fails to write fails its consumer's poll: the
health word). Partials / chunk stats compared over live chunks only.
