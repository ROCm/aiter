I found seven places where FlyDSL kernels issue gfx942-shaped MFMA on gfx950, plus one FP8 decode path that looks like a deliberate trade-off. None of them are correctness bugs. In each case gfx950 runs the `16x16x16` BF16 instruction where the `16x16x32` version would do the same work in half as many instructions. In four of them the code already contains a K=32 path or helper that could be reused. All other gfx950 code I traced already uses native shapes, and there is no `xf32` or FNUZ fp8 reachable on gfx950.

This was a static read only; nothing was compiled or benchmarked. Expected gains are lower on memory-bound decode kernels than on prefill or GEMM kernels.

## Strongest candidates (gfx950 code that uses K=16)

1. **`kernels/mqa_logits/qsa_flydsl_layout.py:217`** (QSA indexer prefill scorer). It is dispatched on both gfx942 and gfx950 and always uses `MFMA(16, 16, 16, BFloat16)` with no arch switch. Its docstring says Q and K share a head-dim permutation and each lane already reads 32 contiguous elements. Two K=16 steps therefore map directly onto one K=32 instruction on gfx950. This is prefill and likely compute-bound, so it is probably the largest real win.

2. **`kernels/gdr_prefill/gdn_fused_prefill.py`**. This kernel is gfx950-only, yet most of its matrix work goes through `_mfma16` (`mfma_f32_16x16x16bf16_1k`):
   - Lines 701 and 791: loops that sum four K=16 blocks into one accumulator. These could be two K=32 instructions, giving A and B the same K permutation.
   - `_mm16_x3` (lines 119–123): `a_lo·b_hi + a_hi·b_lo` could be fused into one K=32. This changes the order in which the hi/lo precision terms are summed, so it would need a numerical check.
   - Lines 685–687: the triangular solve has short dependent loops, so only some of it can be paired.

3. **`kernels/pa_decode` with BF16 KV, single M-tile** (the common decode case). QK goes through `gemm.qk` and `stream_qk` into `mfma()`, which takes the `PV_BF16` path and issues `16x16x16bf16_1k`. The helper `mfma_bf16_k32` already pairs K=16 packs into K=32 MFMAs. It is used only when `BF16_MTP_FUSED` holds, which requires 2–4 M-tiles, head dim 128 and page 128. Decode is memory-bound, so the gain may be small.

4. **`kernels/pa_decode/pipeline.py:1568–1573`** (MXFP4 PV, gfx950-only). It issues one K=16 MFMA per 16-token chunk. The existing `pv_bf16_k32` helper already pairs chunks this way for BF16, so it could likely halve the PV instruction count, provided the staged V layout pairs the same way.

## Medium (deliberate layout choices that could still be improved)

5. **`kernels/qsa/sparse_gqa.py:590`** and **`kernels/qsa/sparse_decode_attn.py:330`**. QK switches to K=32 on gfx950, but PV stays at K=16. The `sparse_decode_attn.py` docstring says why: the QK accumulator layout is exactly the K=16 B operand, so P never leaves registers. Concatenating P from two 16-token tiles, with V permuted to match, keeps that property and uses K=32. This is the same technique `pa_decode` already uses.

6. **`kernels/gdr_prefill/gdn_prepare.py`** (lines 46, 82 and 356). It runs on both gfx942 and gfx950 with no arch branch and uses only K=16. Its `16x16x4` fp32 MFMA is fine: gfx950 has no faster fp32 option, since `xf32` was removed.

## Low (intentional)

7. **`kernels/gdr_prefill/chunk_gated_delta_h.py:300`**. Its docstring says it deliberately matches the hand-tuned HIP kernel's instruction and lane mapping. Changing it means re-deriving the layout.

8. **`pa_decode` FP8 paths on gfx950**. Most use `16x16x32_fp8_fp8`, the gfx942 shape. The wide `16x16x128_f8f6f4` instruction with unit scales is enabled only for `PER_TOKEN_WIDE` and `D256_PIPE`. That looks like a per-shape choice that someone measured, not an oversight.

## Checked and already correct

- **Already K=32 BF16 on gfx950:**
  - GEMM kernels: `preshuffle_gemm`, `gemm_a16w16_gfx950` (plus skinny and band), `gemm_a16wfp4`.
  - MoE: `moe_2stage_a16wmix` `gemm1` and `gemm2`, both correctly guarded to K=16 on gfx942 only.
  - Others: `hstu`, `hyper_connection_gated_residual`, `conv3d`, `minimax_m3_index_score`, `mega_mhc`, `hc_mix`.
- **FP8:** `fp8_mqa_logits` uses FNUZ K=32 on gfx942 and switches to the scaled `MFMA_Scale` K=128/K=64 instructions with OCP fp8 on gfx950.
- **int8:** `gemm_a8w8_int8_4wave` uses `v_mfma_i32_16x16x64_i8`, which is gfx950-native.
- **FNUZ-only kernels are gfx942-only:** `moe_gemm_2stage/gfx9_fp8`, `fmha_bwd_gfx942` and the QSA sparse prefill kernels are only dispatched on gfx942, so they don't affect gfx950.

## Not covered

Some kernels compute the MFMA K from config tables, and I didn't trace each one: `mega_moe*`, `mixed_moe_gemm_2stage_common`, `mxmoe_gemm_v2` and the `mqa_logits` fp4 kernels. Their identifiers mostly point to the scaled `16x16x128` and `32x32x64` instructions, which are gfx950-native.

One thing outside MFMA came up: `linear_attention_prefill_kernels.py:82` sets gfx95 LDS to 128 KB in its fallback table, but gfx950 has 160 KB.

To confirm the top four, I'd compile them for gfx950 without a GPU (`COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx950 FLYDSL_DUMP_IR=1`). Counting the MFMA instructions in the dumped assembly would check this static reading. Prototyping the K=32 pairing for `qsa_flydsl_layout` and checking it with its `op_tests` would follow from that.