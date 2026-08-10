# MoE a16wi4: replace the old int4 2-stage GEMM with the shared a16w-mix port

Routes a16wi4 (bf16 A × int4 W) through the merged a16w-mix port
(`moe_2stage_a16wmix`, `w_dtype="int4"`) instead of the bespoke `int4_bf16`
kernel in `moe_gemm_2stage.py`, which is now dead and deleted (**−3590 LOC**).
a16w4 (mxfp4) already shipped on this kernel in #4502; a16wi4 was its last
remaining separate implementation.

## Perf vs the kernel being replaced

gfx950, 7168×512, E384/topk8. `us_stage1` (gemm1), interleaved candidate/baseline
in one session, min of 3. Ratio < 1.0 = faster than the old kernel.

| tokens | 1 | 4 | 16 | 64 | 512 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|
| **gemm1** | 1.067 | 1.046 | **0.871** | **0.951** | **0.918** | **0.859** | 1.022 |

gemm2 also improves (0.88–0.90 at tok16/64) from the scale-layout change below.
Spot-checked independently in a separate session: tok1 0.980, tok4 1.036,
tok4096 1.027 — agrees within box noise.

Three changes produced this:

1. **Decode tiles.** The port has no grid split-K, so decode is
   workgroup-count-limited: grid = `m_blocks × (inter_dim / tile_n)`. At tok1 the
   inherited tiling gave ~64 workgroups on 256 CUs and gemm1 was **2.58× slower**
   than the old kernel. Narrow N-tiles fix it: 41.6 → 27.6 → **16.7 µs**
   (`tile_n` 64→32→16 with matching `k_wave`).
2. **Rolled K loop (int4 only).** The fully-unrolled body (12747-line ISA,
   448 MFMA) cost 2.0× GRBM and 2.3× SQ_BUSY vs the old kernel's rolled loop *at
   an identical tile*. Rolling it into a loop-carried `scf.for` → ~2200 lines.
   Gated `_is_int4 and BM <= 16`: it wins where the grid is latency-bound and
   loses ~12% at large M, where the kernel is HBM-adjacent and wants the unroll's
   ILP.
3. **Coalesced int4 scale gather.** Lanes 0–15 read 16 consecutive N columns, so
   the N-major scale layout strided them across 16 cache lines per K32 step.
   Switching to the old kernel's `(E, G//2, N, 2)` makes it one 64 B line.

## Caller contract

| | old kernel | this PR |
|---|---|---|
| weight | `pack_int8_to_packed_int4(shuffle_weight(int8,(16,16)))` | **changed** — `shuffle_weight_a16wi4` |
| scale | `shuffle_scale_for_int4` → `(E, G//2, N, 2)` | unchanged |

The weight layout cannot be aligned cheaply. Both preps call the same
`shuffle_weight(...,(16,16))`; the difference is *pack order*, and that sets
MMA-visible K-per-lane. Pack→shuffle makes `KPack=16` count **bytes** = 32
nibbles = K32/lane, so one `dwordx4` feeds one `16x16x32` MMA. Shuffle→pack makes
it count **int8 elements** = 16 nibbles = K16/lane (built for the old kernel's
`16x16x16`), so a K32 fragment needs two `dwordx2` from slots 128 B apart.
Consuming the old layout is implemented and correct but costs ~7–9% at tok1–4,
where the kernel is load-latency-bound. `use_k16` (the gfx942 fallback) is *not*
a shortcut — it only splits the MMA and leaves B-load granularity alone
(verified: forcing it on gfx950 with kpack=8 weights is still correct, and slower).

## Also in this PR

- **AOT**: int4 precompiles through `_precompile_a16w4_to_cache` (the folded
  port's raw-`data_ptr` launchers), so AOT builds the kernel the runtime launches.
- **Dispatch**: the per-token tile table in `get_2stage_cfgs` is gone — tiles are
  a tuning result and come from the CSV. The no-tuned-row path is one shape-safe
  config plus the same warning the mxfp4 fallback emits.
- **Guard**: a16wi4 now raises if FlyDSL is unavailable rather than falling
  through to a CK/ASM stage1, which would consume port-preshuffled weights and
  return wrong numbers instead of failing.
- **Tuned CSV**: retuned decode rows; added `token=16384` rows (previously fell
  to the untuned fallback).

## Validation (gfx950)

- 82/82 CSV-driven int4 cases pass; max `logits_diff` **1.75e-5** vs the old
  kernel's 1e-5–4e-5, i.e. equal or better accuracy.
- Manual sweep tok{1,2,4,16,128,2048,16384} × inter{256,512,1024}: 1.4–1.6e-5.
- **a16w4 (mxfp4) is provably untouched**: gemm1 and gemm2 final ISA are
  byte-identical to `origin/main` (4956 / 716 lines, zero diff). Every change is
  behind `const_expr(_is_int4)`. a16wfp4 pytest 12/12.

Two pre-existing failures show up when running the suite and are **not** from
this PR — `origin/main` reproduces both identically:
- `ck_moe_2stages checkAllclose atol/rtol=0.01 failed` — a tolerance artifact at
  these bf16 magnitudes; gate on `logits_diff`.
- Two a16w4 mxfp4 cases (E257/k9/3072, "2stage default") at `logits_diff ≈ 0.99`.

## Large batch

At **tok16384** (7168 x 512, E384/k8) gemm1 is **2743 us vs the old kernel's 3970 us
(0.69x)** and gemm2 is at parity (1988 vs 2005). Confirmed with rocprofv3 raw
counters at the identical tile: gemm1 GRBM cycles 49.8M vs 75.7M (0.66x) at the
same wave count and grid, with markedly lower VGPR pressure (104 vs 248).
Tuned `token=16384` rows are included; previously this shape fell through to the
untuned fallback.
