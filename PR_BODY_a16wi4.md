# MoE a16wi4: replace the old int4 2-stage GEMM with the shared a16w-mix port

Routes a16wi4 (bf16 A × int4 W) through the merged a16w-mix port
(`moe_2stage_a16wmix`, `w_dtype="int4"`) instead of the bespoke `int4_bf16` kernel
in `moe_gemm_2stage.py`, which is now dead and deleted (**−3590 LOC**). a16w4
(mxfp4) already shipped on this kernel in #4502; a16wi4 was the last separate
implementation.

**Drop-in: the caller contract is unchanged.** Weight and scale preshuffles are
byte-identical to the replaced kernel (`pack_int8_to_packed_int4(shuffle_weight(
w.int8, (16,16)))` and `shuffle_scale_for_int4`), so no model or serving code
changes.

## Perf vs the kernel being replaced

gfx950, 7168×512, E384/topk8. `us_stage1` (gemm1), interleaved candidate/baseline
in one session, min of 3. Ratio < 1.0 = faster.

| tokens | 1 | 4 | 16 | 64 | 512 | 2048 | 4096 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| **gemm1** | 1.03 | 1.02–1.07 | **0.85–0.94** | **0.92–0.97** | **0.94** | **0.86** | 1.02–1.04 | **0.69** | **0.68** |

gemm2 is at parity or better throughout (0.88–1.02). At tok16384 the gemm1 result
is corroborated by rocprofv3 raw counters at the identical tile: GRBM cycles 49.8M
vs 75.7M (0.66×) at the same wave count and grid, with much lower VGPR pressure
(104 vs 248).

Four changes produced this:

1. **Decode tiles.** The port has no grid split-K, so decode is
   workgroup-count-limited: grid = `m_blocks × (inter_dim / tile_n)`. At tok1 the
   inherited tiling gave ~64 workgroups on 256 CUs and gemm1 was **2.58× slower**
   than the old kernel. Narrow N-tiles fix it: 41.6 → 27.6 → **16.7 µs**.
2. **Rolled K loop (int4 only).** The fully-unrolled body (12747-line ISA) cost
   2.0× GRBM and 2.3× SQ_BUSY vs the old kernel's rolled loop *at an identical
   tile*. Rolling it into a loop-carried `scf.for` → ~2200 lines. Gated on
   `BM <= 16`: rolling wins where the grid is latency-bound and costs ~12% at
   large M, which is HBM-adjacent and wants the unroll's ILP.
3. **Coalesced int4 scale gather.** Lanes 0–15 read 16 consecutive N columns, so
   the port's N-major scale layout strided them across 16 cache lines per K32
   step. The old kernel's `(E, G//2, N, 2)` makes it one 64 B line — this is both
   the compatible layout and the faster one.
4. **b_nt=0 at small M.** With kpack=8 the B load is two `dwordx2` per K32
   fragment, and the doubled request count benefits from L2 reuse that streaming
   gave up.

## Also in this PR

- **AOT**: int4 precompiles through `_precompile_a16w4_to_cache` (the folded
  port's raw-`data_ptr` launchers), so AOT builds the kernel the runtime launches.
- **Dispatch**: the per-token tile table in `get_2stage_cfgs` is gone — tiles are
  a tuning result and come from the CSV. The no-tuned-row path is one shape-safe
  config plus the same warning the mxfp4 fallback emits.
- **Guard**: a16wi4 raises if FlyDSL is unavailable rather than falling through to
  a CK/ASM stage1, which would consume port-preshuffled weights and return wrong
  numbers instead of failing.
- **Correctness guard**: the rolled loop is disabled for `k_wave==1 &&
  num_acc_n==1 && tile_k>=256`, where it computes wrong results (an iter-arg/A-DMA
  interaction at that one shape). No production row hits it today, but a future
  retune could pick one.
- **Tuned CSV**: retuned decode rows; added `token=16384` and `32768` rows, which
  previously fell through to the untuned fallback.

## Validation (gfx950)

- 82/82 CSV-driven int4 cases pass; max `logits_diff` **1.75e-5** vs the old
  kernel's 1e-5–4e-5, i.e. equal or better accuracy.
- Manual sweep tok{1,2,4,16,128,2048,16384,32768} × inter{256,512,1024}: 1.4–1.6e-5.
- **a16w4 (mxfp4) is provably untouched**: gemm1 and gemm2 final ISA are
  byte-identical to `origin/main` (4956 / 716 lines, zero diff), re-verified after
  every kernel change. Everything is behind `const_expr(_is_int4)`. a16wfp4
  pytest 12/12.

Two pre-existing failures appear when running the suite and are **not** from this
PR — `origin/main` reproduces both identically:
- `ck_moe_2stages checkAllclose atol/rtol=0.01 failed` — a tolerance artifact at
  these bf16 magnitudes; gate on `logits_diff`.
- Two a16w4 mxfp4 cases (E257/k9/3072, "2stage default") at `logits_diff ≈ 0.99`.

## Notes for reviewers

- `gemm1.py` carries three mainloop variants (single-tile, unrolled, rolled),
  selected at compile time per tile config. The unrolled path is unchanged from
  #4502 and is what mxfp4 always takes.
- The rolled loop emits `scf.ForOp` directly rather than using the `range(...,
  init=)` DSL sugar: `_gemm1_body_a16w4` is a plain helper, not `@flyc.kernel`, so
  the AST rewriter never runs on it. This is a deliberate boundary.
