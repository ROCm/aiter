I found two gfx950 features that are clearly being left on the table in FlyDSL kernels, plus a few smaller issues. The two clear ones:

- **`hstu_attention_fwd`** builds a transposed copy of V in LDS one element at a time, instead of using gfx950's hardware transposed LDS reads.
- **`pa_decode`'s BF16 path** caps LDS at 64 KiB on gfx950 too, where 160 KiB is available.

Everything else I checked already uses the gfx950 feature or has a documented reason not to. Same caveats as the MFMA audit: this was a static, read-only pass over FlyDSL, with nothing compiled or benchmarked.

## Worth acting on

1. **`kernels/hstu/hstu_attention_fwd.py`, lines 222–225 and 609–620.** V is written into LDS already transposed (`[d, kv]`), using per-element stores:

```615:620:aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_V):
                    # transpose scatter: d-contiguous lane elems land at kv-fixed, d-varying slots.
                    v_smem[v_load_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]
```

   That is 8 narrow LDS writes per vector. On gfx950, V could be stored row-major with wide 16-byte writes or a direct-to-LDS load, then read with `ds_read_b64_tr_b16`. Other kernels here already do this through `fx.rocdl.cdna4.LDSReadTrans16_64b()`: `gemm_a16w16_gfx950`, `gdn_fused_prefill` and `sparse_decode_attn`. hstu also runs on gfx942, which lacks the transposed read, so this needs an arch branch. The module docstring (line 9) says V stages "without a transposed LDS scatter", which contradicts the code.

2. **`kernels/pa_decode/traits.py:30`.** `BF16_LDS_LIMIT_BYTES = 64 * 1024` applies on both architectures. On gfx950 it forces more query splits than LDS requires. The `pa_decode_kernel.py` docstring gives examples: D512/MTP4/GQA16 splits twice and D1024/MTP3/GQA16 three ways. The same file already uses `MXFP4_LDS_LIMIT_BYTES = 160 * 1024` for gfx950. The 64 KiB cap might be deliberate to keep two blocks resident per compute unit, but two 80 KiB blocks would still fit in 160 KiB. This needs a benchmark to decide.

## Low priority

3. **`kernels/gdr_prefill/chunk_gated_delta_h.py:33`.** It converts to bf16 by truncation by default (`_BF16_CONVERT_TRUNC_DEFAULT = True`) to match the HIP kernel bit for bit. gfx950 has hardware round-to-nearest-even (`v_cvt_pk_bf16_f32`) at the same cost. This is an accuracy-versus-parity decision, not a performance issue.

4. **`mxscale_preshuffle_kernels.py:40`.** `_GFX950_CUS_PER_XCD = 32`, used with `num_tiles // 8` at line 349, assumes 8 chiplets (XCDs) of 32 compute units each. That holds on a full MI355X but not when the GPU is partitioned, where one device can be a single XCD. `gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py:70` hardcodes `_GFX950_CU_NUM = 256` the same way, though only for tuning.

5. **`linear_attention_prefill_kernels.py:82`.** Its fallback table says gfx95 has 128 KiB of LDS instead of 160 KiB. It's only used when torch doesn't report shared memory, and it errs on the safe side.

6. **`kernels/minimax_m3_index_score_prefill.py:119`.** This gfx950-only kernel sizes its query tile against a self-imposed 64 KiB budget. It does check the real 160 KiB limit afterwards, so the choice looks deliberate.

## Checked and already correct

- **LDS capacity:** most launchers call `get_lds_capacity_bytes(arch)`. The hardcoded 64 KiB limits are in gfx942-only code (`fused_moe_gfx942`, `moe_gemm_2stage/gfx9_fp8`).
- **Direct-to-LDS width:** gfx950 operand loads use 16 bytes per lane (`BufferCopyLDS128b`, or `raw_ptr_buffer_load_lds` with size 16). The 1- and 4-byte copies are only used for scale gathers, where the layout requires them. `moe_2stage_a16wmix` correctly falls back to staging through registers on gfx942.
- **bf16 conversion:** the hand-written rounding (`0x7FFF` bit tricks) appears only in gfx942-only code (`gfx9_fp8/common.py`, `qsa/sparse_prefill_common.py`). gfx950 paths use `.to(BFloat16)` or `cvt_pk_bf16_f32`.
- **fp4/fp8 quantization:** the gfx950 MX paths use the scaled conversion instructions (`cvt_scalef32_*`) widely, in about 35 files.
- **Lane swaps:** `permlane16_swap` and `permlane32_swap` are already used in `pa_decode`, `fmha_gfx950`, the fp4 MQA-logits kernels, the blockscale GEMM epilogue, `mega_moe_tp` and `sparse_pa_nvfp4`.
- **CU counts:** these come from `get_cu_num()` or the device properties. The literal 256 fallbacks are in gfx1250 code, and `gemm_a16w16_band` explicitly requires 256 compute units.

## Not covered

- **Lane-swap opportunities:** finding `ds_bpermute` or shuffle sequences that could become `permlane*_swap` needs per-kernel reasoning, which grep can't do.
- **Register staging:** kernels that stage through registers instead of direct-to-LDS loads (for example the `pa_decode` and QSA kernels) look like design choices, so I didn't flag them.
- **fp8 quantization guards:** the guards before `cvt_pk_fp8_f32` in `fused_compress_attn` and `qk_norm_rope_quant` were written with gfx942's FNUZ behaviour in mind. They may be unnecessary on gfx950, but I didn't verify that.
- **Tuned config CSVs:** outside the scope of this pass.

The natural next step is to prototype the hstu transposed-read change on gfx950 and benchmark the `pa_decode` BF16 LDS cap at 64 KiB against 160 KiB with the existing `op_tests`.