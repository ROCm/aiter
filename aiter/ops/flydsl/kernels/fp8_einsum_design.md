# FP8 Einsum Kernel — Design (plan v3)

Implements a DeepGEMM-equivalent `fp8_einsum('bhr,hdr->bhd', x, y)` for AMD
gfx950 in FlyDSL. Lives in `fp8_einsum.py`, structurally cloned from
`compile_preshuffle_gemm_a8` in `preshuffle_gemm.py`.

## Contract

```python
compile_fp8_einsum_bhr_hdr_bhd(
    *, H, D, R,                  # compile-time
    tile_m, tile_n, tile_k,      # tile_k % 128 == 0; tile_m in {16,32,64,128,256}
    scale_mode: str = "fp32",    # "fp32" | "ue8m0"
    lds_stage=2, waves_per_eu=None,
    use_async_copy=False, dsrd_preload=-1, dvmem_preload=-1,
    xcd_swizzle=0,
) -> launch_fn

# launch_fn(arg_z, arg_x, arg_y, arg_sy, i32_b, stream)
#   arg_z : bf16 (B, H, D)
#   arg_x : bf16 (B, H, R)                 ← bf16, NOT pre-quantized
#   arg_y : fp8 e4m3, preshuffled per head, layout (n0, k0, 4, 16, 16)
#   arg_sy: fp32  (H, D//128, R//128)            if scale_mode="fp32"
#           int32 (H, D//128, R//(128*4))        if scale_mode="ue8m0" (4 UE8M0 per int32)
#   i32_b : runtime batch size B
```

No `arg_sx`: activation scale is computed online inside `compute_tile` from
the bf16 `x`. This is the only deliberate divergence from DeepGEMM's contract,
which expects the caller to pass pre-quantized FP8 + scales.

## Why "online activation quant, offline weight quant"

Mirrors the production deployment pattern: weights are quantized once at load
time (offline, cheap), activations are quantized on every forward pass (must
be fused to be cheap). DeepGEMM punts the activation cast to a separate kernel
and asks users to fuse it into the previous op (RMSNorm/dispatch/...). Here we
fuse it directly into the GEMM instead.

## Mapping einsum → GEMM workgroup grid

`Z[b, h, d] = Σ_r X[b, h, r] · Y[h, d, r]`. Treated as `H` independent
`(M=B, N=D, K=R)` GEMMs sharing weights only within an `h`.

Grid (mirrors DeepGEMM `m_grouped_fp8_gemm_*_contiguous`):

```
gx = ceil(B, tile_m) * H            # outer: h, inner: m-tile within b
gy = D // tile_n
gz = 1
```

`bx_h = bx // ceil_div(B, tile_m)` (constant within a workgroup) — folded into
the Y base pointer once. `bx_m = (bx % ceil_div(B, tile_m)) * tile_m`.

Strides: `H, D, R` are compile-time, so all of `(stride_b_x=H*R, stride_h_x=R,
stride_b_z=H*D, stride_h_z=D, stride_h_y=D*R)` are constants. Buffer-resource
`num_records_bytes` for X is `B * H * R * 2` (bf16).

## The two scale modes — compile-time switches

| | `scale_mode="fp32"` | `scale_mode="ue8m0"` |
|---|---|---|
| MFMA | `mfma_f32_16x16x32_fp8_fp8` (K=32) | `mfma_scale_f32_16x16x128_f8f6f4` (K=128) |
| MFMAs per 128-K | 4 | 1 |
| Scratch acc | Yes, reset per 128-K, promoted to global_acc | **No** — MFMA folds scales |
| s_y storage | fp32, 1 per (n_block, k_block) | UE8M0, 4 packed in int32 |
| s_x compute | `clamp(amax,1e-4) * (1/448)` | `ceil_to_ue8m0(clamp(amax,1e-4) * (1/448))` |
| s_x in cast | true `v_div_f32` for `1/s_x` then RNE cast | pow-of-2: `inv_s = bits((254-exp)<<23)`, multiply, RNE cast |
| Scale apply | `global_acc += scratch * (s_x * s_y)` in fp32 | hardware-fused inside MFMA via the scale operands |

The switch lives inside `compute_tile`. All other paths (LDS pipeline, B load,
block-id decomposition, output store) are shared.

## `compute_tile` algorithm (per K-tile = `tile_k / 128` quant groups)

```text
for k128 in range(tile_k // 128):
    # Phase 1: bf16 from LDS, promote to fp32
    a_bf16 = lds_load_bf16_chunk(...)               # per-lane fragment, K=128/lane_group
    a_fp32 = bf16_to_fp32(a_bf16)

    # Phase 2: per-row amax across all 128 K-elements
    local  = max(|a_fp32|, axis=k_in_lane)          # intra-lane reduction
    final  = dpp_butterfly_xmask_1_then_2(local)    # cross 4 lane_div_16 groups
    amax   = clamp(final, 1e-4)

    if scale_mode == "fp32":
        s_x_f32   = amax * (1.0/448.0)
        inv_s_f32 = 1.0 / s_x_f32                   # true v_div_f32
        a_fp8     = cvt_pk_fp8_f32(a_fp32 * inv_s_f32)
        # 4x MFMA into scratch (K=32 each)
        scratch[mi,ni] = 0
        for ku in range(4):
            scratch[mi,ni] = mfma_fp8(a_fp8_chunk, b_fp8_chunk, scratch[mi,ni])
        # Promote into global
        s_y_f32 = load_s_y_f32(...)                 # 1 per ni-block
        global_acc[mi,ni] += scratch[mi,ni] * (s_x_f32 * s_y_f32)

    elif scale_mode == "ue8m0":
        bits      = bitcast<i32>(amax * (1/448))
        exp_raw   = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1 : 0)
        exp       = clamp(exp_raw, 1, 254)          # 8-bit UE8M0
        inv_s_f32 = bitcast<f32>((254 - exp) << 23) # = 2^-(exp-127) as fp32 pow2
        a_fp8     = cvt_pk_fp8_f32(a_fp32 * inv_s_f32)
        # Pack 4 row-exponents into i32 for MFMA scale operand
        s_x_i32 = pack_4_row_exponents(exp)         # per lane-group
        s_y_i32 = load_s_y_packed_ue8m0(...)        # 1 i32 = 4 K-blocks of scales
        # One scaled MFMA — hardware folds in scales
        global_acc[mi,ni] = mfma_scale_f32_16x16x128_f8f6f4(
            a_fp8, b_fp8, global_acc[mi,ni],
            cbsz=0, blgp=0, abid=0,
            s_x_i32, /*scale_block_sel_b=*/0, s_y_i32,
        )
```

## Cross-lane amax (the new primitive)

Layout per wave: `row = lane_mod_16`, `k_group = lane_div_16 ∈ {0,1,2,3}`.
For a 128-K chunk, each lane holds 32 fp8 elements covering 4 stripes of 8 K
(= 4 MFMA calls' worth for the K=32 MFMA). After local amax (32→1 in
registers), the 4 lanes sharing a `row` must agree:

```python
local_amax  = max_reduce(|a_fp32_frag|)             # fp32
v1          = update_dpp(local_amax, row_xmask:1)   # swap with lane ^ 16
m1          = max(local_amax, v1)
v2          = update_dpp(m1, row_xmask:2)           # swap with lane ^ 32
final_amax  = max(m1, v2)
```

Two `v_mov_b32_dpp` + two `v_max_f32`. `max` is associative + exact so
ordering is irrelevant for accuracy.

The actual rocdl intrinsic is `rocdl.update_dpp` with `dpp_ctrl` literals
`0x101` (`row_xmask:1`) and `0x102` (`row_xmask:2`); cross-lane masks
`row_mask=0xF`, `bank_mask=0xF`, `bound_ctrl=true`.

## Accuracy alignment with DeepGEMM (bit-level)

### FP32 mode = DeepGEMM `use_ue8m0=False`
- amax in fp32 (not bf16). `bf16->fp32` promote before reduction.
- `clamp(amax, 1e-4)` before dividing by 448.
- true `v_div_f32` for `1/s_x`, not `v_rcp_f32` (~1 ULP).
- `cvt_pk_fp8_f32` with RNE (default mode), no stochastic rounding.
- Scale-apply order: `scratch * (s_x * s_y)` — multiply scales together first.
- Validates against:
  ```python
  sx = (x_amax / 448).clamp_min(1e-4)
  x_fp8 = (x / sx).to(fp8_e4m3)
  sy, y_fp8 = per_block_cast_to_fp8(y, use_ue8m0=False)
  z_ref = einsum_fp32_path(x_fp8, sx, y_fp8, sy)
  assert calc_diff(z_kernel, z_ref) < 1e-3
  ```

### UE8M0 mode = DeepGEMM `use_ue8m0=True`
- Same amax + clamp + /448 as FP32 mode.
- Then `ceil_to_ue8m0` bit-trick → power-of-2 scale, ≤ 1-bit mantissa loss in FP8.
- `inv_s` is a pow-of-2 — no division, just an exponent decrement.
- `cvt_pk_fp8_f32` with RNE (same as FP32 mode).
- Hardware folds `s_x * s_y` inside `mfma_scale_f32_16x16x128_f8f6f4`.
- Validates against `per_token_cast_to_fp8(use_ue8m0=True)` + `per_block_cast_to_fp8(use_ue8m0=True)`.

## What we deliberately do NOT do (to match DeepGEMM)

- No Kahan / compensated summation.
- No higher-precision FP8 cast — use the hardware RNE intrinsic.
- No re-quantization beyond what each mode dictates.
- No fused bias / activation in v1.

## LDS sizing impact

`elem_bytes_a = 2` (bf16) vs template's 1 (fp8). So:
- `bytes_a_per_tile = tile_m * tile_k * 2`
- LDS for A doubles vs the fp8-A template
- `bytes_per_thread_a` doubles → `num_a_loads` doubles for the same tile
- B path is unchanged (preshuffled fp8, 1 byte/elem, `load_b_pack_k32`)
- LDS stage 2 ping-pong same as template, just larger buffer per stage

For `tile_m=64, tile_k=128`: A occupies 16 KiB per stage × 2 stages = 32 KiB.
Plenty of headroom on gfx950 (160 KiB LDS).

## Build sequence (tracked in TaskList)

1. **Skeleton + bf16 LDS pipeline** — clone template, strip non-FP8, wire
   `(h, m_tile)` grouping, validate with `mfma_f32_16x16x16bf16_1k` (no quant).
2. **Cross-lane amax** — standalone test of DPP butterfly.
3. **`scale_mode="fp32"` end-to-end** — full compute_tile per §"compute_tile algorithm".
4. **`scale_mode="ue8m0"` end-to-end** — swap 4 switches, reuse rest.
5. **Perf sweep** — `(tile_m, tile_n, tile_k) ∈ {64,128,256}×{128,256}×{128,256}`.

## Test target

Port DeepGEMM's `test_fp8_bhr_hdr_bhd`: shapes `(h=8, r=4096, d=1024)`,
`b ∈ {4, 32, 128, 4096, 8192}`. Tolerance: `calc_diff < 1e-3`.

## Open items deferred to v2

- General einsum strings (only `bhr,hdr->bhd` in v1).
- bf16 weight path (only preshuffled fp8 Y in v1).
- gfx942 support (gfx950 only in v1).
- Fused bias / activation epilogue.
- Plain (non-preshuffled) B layout.

## Key reference points in `preshuffle_gemm.py`

- Entry point: `compile_preshuffle_gemm_a8` (L115)
- Main kernel: `kernel_gemm` (L346)
- A DMA helper: `dma_a_tile_to_lds` (L736)
- A LDS load: `lds_load_packs_k64` (L654)
- B preshuffle layout + load: L370-380, `load_b_tile` (L610)
- Compute tile (template for ours): `compute_tile` (L901)
  - FP32-mode template ≈ the default branch (L1091-1151)
  - UE8M0-mode template ≈ the FP4 branch (L939-1038) with FP8 element format
- Epilogue: `store_output` (L1154)
- Launcher: `launch_gemm` (L2133)
