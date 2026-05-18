# FP8 Einsum Kernel — Handoff Notes

Status snapshot for the next agent picking up `fp8_einsum('bhr,hdr->bhd', x, y)`
on AMD gfx950 in FlyDSL. Reads alongside `fp8_einsum_design.md` (the spec) and
`fp8_einsum.py` (the implementation).

## TL;DR — what's done, what isn't

| Item | Status | Where |
|---|---|---|
| Skeleton + bf16 LDS pipeline + (h,m_tile) grouping | ✅ done | `fp8_einsum.py` |
| Cross-lane per-row amax (DPP butterfly) | ✅ done | `cross_lane_amax` |
| `scale_mode="fp32"` end-to-end compute_tile | ✅ done | `compute_tile_fp32` |
| Unit test: compile trace (any host) | ✅ done | `test_fp8_einsum.py` |
| Unit test: end-to-end accuracy (gfx950 only) | ✅ done, **unrun** | same |
| `scale_mode="ue8m0"` | ❌ stubbed (`NotImplementedError`) | `compile_fp8_einsum_bhr_hdr_bhd:73` |
| Perf sweep / autotune | ❌ pending | — |
| LDS ping-pong (`lds_stage=2`) | ❌ pending (single-buffer only) | `compile_fp8_einsum_bhr_hdr_bhd:151` |

**Cannot run accuracy test locally** — implementation was done on a gfx942 host;
the kernel guards `gpu_arch.startswith("gfx95")` and the MFMA intrinsics
(`mfma_f32_16x16x32_fp8_fp8`, `mfma_scale_f32_16x16x128_f8f6f4`) require gfx950.

## Files

- `aiter/ops/flydsl/kernels/fp8_einsum.py` — kernel + launcher
- `aiter/ops/flydsl/kernels/fp8_einsum_design.md` — full design / accuracy contract
- `aiter/ops/flydsl/kernels/fp8_einsum_handoff.md` — this doc
- `op_tests/flydsl_tests/test_fp8_einsum.py` — pytest suite

## Contract (unchanged from design doc)

```python
compile_fp8_einsum_bhr_hdr_bhd(
    *, H, D, R,                  # compile-time
    tile_m, tile_n, tile_k,      # tile_k % 128 == 0; tile_m % 16; tile_n % 64
    scale_mode: str = "fp32",    # "fp32" | "ue8m0"  (ue8m0 stubbed)
    lds_stage=1, waves_per_eu=None,
    use_async_copy=False, xcd_swizzle=0,
) -> launch_fn

# launch_fn(arg_z, arg_x, arg_y, arg_sy, i32_b, stream)
#   arg_z : bf16 (B, H, D)
#   arg_x : bf16 (B, H, R)             ← bf16, NOT pre-quantized; kernel quantizes online
#   arg_y : fp8 e4m3, preshuffled per head, layout (n0,k0,4,16,16)
#   arg_sy: fp32 (H, D//128, R//128)             if scale_mode="fp32"
#           int32 (H, D//128, R//(128*4))        if scale_mode="ue8m0" (not impl)
#   i32_b : runtime batch size B
```

Mapping einsum → GEMM:
- Treats as `H` independent `(M=B, N=D, K=R)` GEMMs sharing weights only within `h`.
- Grid: `gx = ceil(B,tile_m)*H`, `gy = D//tile_n`, `gz = 1`.
- Per-WG: `bx_h = bx // ceil_div(B,tile_m)`, `bx_m = (bx % ceil_div(B,tile_m)) * tile_m`.

## The fp32-mode algorithm (what `compute_tile_fp32` actually does)

For each `tile_k` chunk, loop over `groups_per_tile = tile_k/128` quant groups.
Per group, per `mi ∈ range(tile_m/16)`:

1. **LDS load + fp32 promote**: 4 × 8-element bf16 chunks (= 32 bf16/lane covering
   the full 128-K group), promoted to fp32 (`a_bf16.to(fx.Float32)`).
2. **Per-lane amax**: `max(|v|)` over those 32 fp32 elements; abs via `max(v, -v)`
   (couldn't confirm `arith.absf` is wrapped — `maximumf` definitely is).
3. **Cross-lane amax** via 2-step DPP butterfly (see below).
4. **Compute s_x**: `s_x = max(amax, 1e-4) / 448`; `inv_s = 1.0 / s_x` (true
   `v_div_f32`, matches DeepGEMM `use_ue8m0=False`).
5. **Quantize**: `(a_fp32 * inv_s)` → `cvt_pk_fp8_f32` (RNE), pack 8 fp32 → 1 i64.
6. **4 × MFMA into scratch**: scratch reset to 0 per group, `mfma_f32_16x16x32_fp8_fp8`.
7. **Load `s_y`** per `ni` for this K-128 group from `arg_sy`.
8. **Promote scratch → global**: per `(ii, ni)`, fetch `s_x` for output row via
   `ds_bpermute_f32(src_lane = lane_div_16*4 + ii, s_x)`, then
   `global[mi,ni][ii] += scratch[mi,ni][ii] * (s_x_for_out * s_y[ni])`.

The scratch-then-promote pattern matches DeepGEMM's SM90 path exactly — scratch
holds the un-scaled dot of fp8 values, and the rescale happens in fp32 register.

## Key implementation pieces (with file:line markers)

### Cross-lane amax — DPP butterfly (`cross_lane_amax`, ~L470)
```python
# lane row layout: row = lane%16, k_group = lane/16 ∈ {0,1,2,3}
# 4 lanes sharing `row` agree on amax via 2 swaps:
#   step 1: row_xmask:1  (dpp_ctrl=0x101) — lanes swap across rows {0↔1, 2↔3}
#   step 2: row_xmask:2  (dpp_ctrl=0x102) — lanes swap across rows {0↔2, 1↔3}
# DPP operates on i32 → bitcast fp32↔i32 around each call.
# row_mask=0xF, bank_mask=0xF, bound_ctrl=True
```
Uses low-level `flydsl._mlir.dialects.rocdl.update_dpp` (the high-level
`fx.expr.rocdl` wrapper may not expose dpp_ctrl literals).

### Per-output-row source-lane mapping (`src_lane_for_ii`, ~L519)
Critical assumption: **MFMA output vec lays out `row_in_tile = lane_div_16*4 + ii`
for `ii ∈ {0,1,2,3}`** (the 4 fp32 lanes of `Vec<4×f32>` returned by
`mfma_f32_16x16x32_fp8_fp8`).

The input quant lane holds `s_x` for `row = lane%16` (with `lane_div_16=0`
producing the canonical copy after the DPP reduce). So to fetch `s_x` for
output row `lane_div_16*4 + ii`:
```python
src_lane = lane_div_16 * 4 + ii    # which input lane (lane%16) holds it
s_x_for_out = ds_bpermute_f32(src_lane, s_x)
```

**If accuracy test fails with permutation-style wrongness (not magnitude), this
mapping is the prime suspect.** Alternative: the MFMA may lay out `ii` as the
column-of-4 instead, in which case the source-lane formula differs.

### `ds_bpermute_f32` (~L498)
Internally multiplies the lane index by 4 (ds_bpermute wants a byte offset
`lane_idx * 4`). Bitcasts fp32 ↔ i32 around the call.

### B preshuffle layout (`load_b_tile`, ~L383)
Per-head: `(n0, k0, klane=4, nlane=16, kpack=16)` fp8.
- `_stride_nlane = 16`
- `_stride_klane = 16 * 16 = 256`
- `_stride_k0 = 4 * 256 = 1024`
- `_stride_n0 = k0_val * 1024` where `k0_val = R // 64`
- Per-head stride into Y: `bx_h * (D * R)` (fp8: 1 byte/elem)

Host-side preshuffle for the test uses `shuffle_weight(y_HDR_fp8, layout=(16, 32))`
— `layout=(IN=16, IK=32)` produces `BN=16, BK=64`, which gives the
`(n0, k0, 4, 16, 16)` layout. See `aiter/ops/shuffle.py:7`.

### Buffer load idioms — easy to get wrong
- `buffer_copy_gmem16_dwordx4` with `elem_bytes=2` (bf16): pass an **element**
  index — the helper does `shrui(idx, 1)` internally for elem_bytes=2. **Do not
  pre-multiply by 2.**
- `buffer_ops.buffer_load(..., dtype=fx.Int32)`: idx is in **dwords**. For Y
  this means `idx_dword = idx_byte // 4`.
- `buffer_ops.buffer_load` has **no** `offset_is_bytes` kwarg (only
  `buffer_store` does).

### LDS swizzle
Same `swizzle_xor16(row, col, k_blocks16)` scheme as
`mfma_preshuffle_pipeline.py:29` — col XOR ((row & (k_blocks16-1)) * 16).
Store and load both use it.

### `cvt_pk_fp8_f32` signature gotcha
```python
rocdl.cvt_pk_fp8_f32(res_type, src_a, src_b, old_i32, word_sel)
#                    positional;  word_sel ∈ {0, 1};  use plain int, not fx.Int1
```
There's no `fx.Int1` — it doesn't exist.

### Compile-time strides
- `stride_h_x_elems = R` (bf16 elems between heads of X within a batch)
- `stride_b_x_elems = H * R`
- `stride_b_z_elems = H * D`
- `_sy_per_head = (D // 128) * (R // 128)` (fp32 elems)
- `_sy_per_n128 = R // 128`

`sy_idx = bx_h * _sy_per_head + n_block_g * _sy_per_n128 + k_block_g` (fp32 elem
index — `buffer_load` with `dtype=fx.Float32` walks fp32 elements).

## Implementation plan (what was followed)

This is the plan the previous agent worked from. Steps 1-3 are done; 4-5 remain.

### Phase 0 — Anchor on a template

Cloned the structure of `compile_preshuffle_gemm_a8` in `preshuffle_gemm.py`
rather than building from scratch. The template already handles:
- workgroup grid + (h, m_tile) decomposition
- buffer-resource creation for A/B/Z
- LDS allocation + swizzled store/load helpers
- B preshuffle global load
- the K-loop driving structure
- the epilogue store via `mfma_epilog`

Key reference points (line numbers in `preshuffle_gemm.py`):
- Entry point: `compile_preshuffle_gemm_a8` (L115)
- Main kernel: `kernel_gemm` (L346)
- A DMA helper: `dma_a_tile_to_lds` (L736)
- A LDS load: `lds_load_packs_k64` (L654)
- B preshuffle layout + load: L370-380, `load_b_tile` (L610)
- `compute_tile` template (with both fp32 and ue8m0 branches): L901
- Epilogue: `store_output` (L1154)
- Launcher: `launch_gemm` (L2133)

The two deliberate divergences from the template:
1. **A is bf16, not fp8** — kernel does online quant inside `compute_tile`.
   This doubles A's LDS bytes (2 bytes/elem vs 1).
2. **Grid encodes `(h, m_tile)` in bx** — template uses bx for m only;
   we fold an extra `h` dim because each head has independent weights.

### Phase 1 — Skeleton (Step 1)

Goal: a kernel that compiles, loads bf16 A into LDS, loads preshuffled fp8 B
into registers, and stores a placeholder output. Validates plumbing without
worrying about correctness of compute.

Tasks done:
1. Compile-time validation block (`tile_k % 128 == 0`, `tile_m % 16`,
   `tile_n % 64`, `R % tile_k`, etc.)
2. Strides table: `stride_b_x_elems = H*R`, `stride_h_x_elems = R`,
   `stride_b_z_elems = H*D`, `_sy_per_head = (D//128)*(R//128)`,
   `_sy_per_n128 = R//128`.
3. Workgroup grid math: `gx_per_h = ceil(B, tile_m)`, `bx_h = bx // gx_per_h`,
   `bx_m = (bx % gx_per_h) * tile_m`.
4. LDS sizing for bf16 A only (B is register-only). `lds_total_elems_a =
   tile_m * tile_k`.
5. Per-head Y byte base: `y_head_byte_off = bx_h * (D * R)` (fp8: byte == elem).
6. `load_a_tile_to_lds` using `buffer_copy_gmem16_dwordx4` (helper does the
   element→byte conversion internally for elem_bytes=2 — don't pre-multiply).
7. `load_b_tile` using `buffer_ops.buffer_load(..., dtype=fx.Int32)` with idx
   in dwords (not bytes — no `offset_is_bytes` kwarg exists for `buffer_load`).
8. `lds_load_bf16_8elems` mirroring the template's swizzled LDS load.

Verify: compile-trace the empty kernel before moving on. (Done.)

### Phase 2 — Cross-lane amax (Step 2)

Goal: a primitive that takes a per-lane fp32 value (the local amax over 32
elements) and returns the max across the 4 `lane_div_16` groups that share a
row.

Lane layout per wave: `row = lane%16, k_group = lane/16 ∈ {0,1,2,3}`. For each
`row`, 4 lanes hold different K-stripes of the same row and need to agree on
the row's amax. Two DPP swaps + maxes suffice:

```
v1   = update_dpp(local, row_xmask:1)   # swap with lane^16  (0↔1, 2↔3 across rows)
m1   = max(local, v1)
v2   = update_dpp(m1,    row_xmask:2)   # swap with lane^32  (0↔2, 1↔3 across rows)
final = max(m1, v2)
```

DPP control literals: `0x101` for row_xmask:1, `0x102` for row_xmask:2.
`row_mask=0xF`, `bank_mask=0xF`, `bound_ctrl=True`. DPP operates on i32, so
bitcast fp32 ↔ i32 around each `update_dpp` call.

Used the low-level `flydsl._mlir.dialects.rocdl.update_dpp` because the
high-level `fx.expr.rocdl` wrapper didn't appear to expose `dpp_ctrl` literals.

### Phase 3 — `scale_mode="fp32"` end-to-end (Step 3)

This is the hard one. The skeleton already loads A (bf16) and B (fp8); now
`compute_tile_fp32` has to:

1. **Iterate over K-128 quant groups** inside the K-tile (`groups_per_tile =
   tile_k // 128`).
2. **Per group, per `mi`**: load 4 × 8-elem bf16 chunks from LDS, promote to
   fp32, compute per-lane amax across the 32 elements, cross-lane reduce,
   clamp + divide by 448, divide 1/s_x (true `v_div_f32`, not `rcp` — matches
   DeepGEMM exactly), then quantize.
3. **4 × MFMA into scratch**: `mfma_f32_16x16x32_fp8_fp8` accumulates into a
   per-`(mi, ni)` scratch register reset to zero per group.
4. **Promote scratch → global**: for each of the 4 output rows the lane holds
   (`ii ∈ {0,1,2,3}`), fetch `s_x` from the input-quant lane that holds it
   (`ds_bpermute_f32`), multiply by `s_y[ni]`, then FMA into the global acc.

#### The row-remap problem (this took the longest to figure out)

Input quant lane distribution: lane with `lane_mod_16 = L` holds the
amax / s_x for input row `L`. After the DPP butterfly, all 4 lanes with the
same `lane_mod_16` agree.

MFMA output distribution (the assumption): each lane writes 4 output rows, and
the 4 elements of the returned `Vec<4×f32>` correspond to rows
`mi*16 + lane_div_16*4 + ii` for `ii ∈ {0,1,2,3}`.

So at output time, the lane currently holding scratch for output row
`lane_div_16*4 + ii` needs `s_x` from input-quant lane `lane_mod_16 ==
lane_div_16*4 + ii`. That's a `ds_bpermute` with `src_lane = lane_div_16*4 + ii`.

`ds_bpermute` byte-vs-dword: the helper treats the index as a byte offset
(`lane_idx * 4`). Implemented `ds_bpermute_f32` to do the `*4` internally so
callers pass a clean lane index.

Considered three alternatives before settling on `ds_bpermute`:
1. **LDS roundtrip** — store all 64 s_x values to LDS, every lane reads what
   it needs. Works but adds barrier + LDS traffic.
2. **`readlane`** — but the source lane varies per destination lane (`ii`
   embeds in the formula), so `readlane` (which uniform-broadcasts) doesn't fit.
3. **`ds_bpermute`** — variable per dest lane, single instruction. Picked this.

#### `s_y` indexing

`arg_sy` layout `(H, D//128, R//128)` fp32:
```python
n_col_global = by_n + n_tile_base + (ni * 16) + lane_mod_16
n_block_g    = n_col_global // 128
k_block_g    = (base_k_elem_in_head + g * 128) // 128
sy_idx       = bx_h * _sy_per_head + n_block_g * _sy_per_n128 + k_block_g
```
Each lane loads its own `s_y[ni]` per ni per K-128 group.

### Phase 4 — `scale_mode="ue8m0"` (Step 4) — DEFERRED

Decided NOT to implement speculatively because of two unknowns about
`mfma_scale_f32_16x16x128_f8f6f4` that need hardware to validate:
1. Layout of the 4 UE8M0 bytes packed in `s_x_i32` — 4 K-blocks per lane? 4
   rows? something else?
2. Whether `s_x` distribution needs a remap (like the fp32 path's
   `ds_bpermute`) or matches the input quant layout natively.

Guessing produces a kernel that silently computes wrong numbers. Stubbed with
`NotImplementedError` and the design-doc algorithm documented for the next
agent.

### Phase 5 — Perf sweep (Step 5) — DEFERRED

Will require `lds_stage=2` ping-pong first (currently `lds_stage=1` single
buffer). Template at `preshuffle_gemm.py` has the ping-pong pattern.

### Phase 6 — Tests

Wrote `test_fp8_einsum.py` with three layers:
1. **Compile-trace** (`test_fp8_einsum_compile_trace`): runs anywhere, monkey-
   patches the arch guard, validates IR build for several tile shapes.
2. **UE8M0 stub guard** (`test_fp8_einsum_ue8m0_mode_not_implemented`): ensures
   the stub stays a stub until intentionally enabled.
3. **End-to-end accuracy** (`test_fp8_einsum_bhr_hdr_bhd_fp32_mode`): gfx950
   only. Self-contained reference (`per_token_cast_to_fp8_x`,
   `per_block_cast_to_fp8_y`, `einsum_fp32_path`, `calc_diff`) that's bit-
   aligned to the kernel's scratch-then-promote scheme.

The end-to-end reference deliberately does the same scratch-per-K128-block
pattern as the kernel — both ref and kernel sum 128 fp32 fmas, then multiply by
`s_x * s_y`, then add to global. This way, FP8 lossy quantization is the only
source of error and tolerance can be tight (`1e-3` cosine-distance).

### Design decisions worth knowing about

- **Online activation quant, offline weight quant** — mirrors production
  deployment. DeepGEMM punts the activation cast to a separate kernel; we fuse.
  Y is the only input that's pre-quantized (offline, once per model load).
- **`tile_m = 1` excluded** — throughput-focused; MFMA 16x16 doesn't degrade
  gracefully to single-row.
- **`lds_stage=1` only in v1** — simpler. Ping-pong is a perf optimization, not
  a correctness one.
- **gfx950 only** — MFMA intrinsics differ on gfx942. No attempt at portability.
- **No fused bias / activation epilogue** — out of scope for v1.
- **No Kahan / compensated sum, no higher-precision FP8 cast** — matches
  DeepGEMM bit-for-bit, no "better-than" math.

## Build sequence — where each step lives

| Step | Description | Status |
|---|---|---|
| 1 | Skeleton + bf16 LDS pipeline + (h, m_tile) grouping | ✅ |
| 2 | Cross-lane amax via DPP butterfly | ✅ (`cross_lane_amax`) |
| 3 | `scale_mode="fp32"` end-to-end | ✅ (`compute_tile_fp32`) |
| 4 | `scale_mode="ue8m0"` end-to-end | ❌ stubbed |
| 5 | Perf sweep `(tile_m, tile_n, tile_k) ∈ {32,64,128,256}³` | ❌ |

## What's left to implement

### Step 4 — UE8M0 mode (high uncertainty, hardware needed)

The blocker: `mfma_scale_f32_16x16x128_f8f6f4` takes two per-lane packed UE8M0
scale operands (each `int32` = 4 UE8M0 bytes). Two unknowns:

1. **Within a lane's `s_x_i32`, do the 4 bytes correspond to 4 successive K-blocks
   (k=0,1,2,3 within the 4 × K=32 chunks of the K=128 MFMA), or to 4 successive
   rows?**
2. **Across lanes, does `s_x` need a row remap (like the fp32 path's
   `ds_bpermute`) or does the hardware consume per-lane scales matching the input
   quant layout `row = lane%16`?**

Without gfx950 in hand, guessing produces a kernel that silently computes wrong
numbers. The recommended path:
- Write a smoke kernel with `s_x = 0x7F7F7F7F` (UE8M0 of 1.0, per
  `preshuffle_gemm.py:1039-1088` placeholder) and `s_y` similarly, then verify
  the result equals unscaled MFMA output.
- Then perturb one byte of `s_x` and observe which rows/columns/K-blocks scale
  — that reveals the layout.

Algorithm (per design doc):
```python
# Per-row amax → s_x via UE8M0 bit-trick (no division):
bits      = bitcast<i32>(amax * (1/448))
exp_raw   = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1 : 0)
exp       = clamp(exp_raw, 1, 254)
inv_s_f32 = bitcast<f32>((254 - exp) << 23)    # = 2^-(exp-127), pow-of-2
a_fp8     = cvt_pk_fp8_f32(a_fp32 * inv_s_f32) # RNE, same intrinsic as fp32 mode

# Pack 4 row-exponents into i32 for MFMA scale operand:
s_x_i32 = pack_4_row_exponents(exp)   # per lane-group — LAYOUT UNKNOWN
s_y_i32 = load_s_y_packed_ue8m0(...)  # 1 i32 = 4 K-blocks of scales

# One scaled MFMA — hardware folds in scales:
global_acc[mi,ni] = mfma_scale_f32_16x16x128_f8f6f4(
    a_fp8, b_fp8, global_acc[mi,ni],
    cbsz=0, blgp=0, abid=0,
    s_x_i32, /*scale_block_sel_b=*/0, s_y_i32,
)
```

`arg_sy` shape change: `int32 (H, D//128, R//(128*4))` (4 UE8M0 packed per int32).

### Step 5 — Perf sweep
Single-buffer `lds_stage=1` only right now. For perf parity with the template,
need to:
1. Wire `lds_stage=2` ping-pong (template at `preshuffle_gemm.py` L901+ shows the
   pattern with `dma_a_tile_to_lds` + `lds_load_packs_k64`).
2. Sweep `(tile_m, tile_n, tile_k)`. Current parametrize covers
   `tile_m ∈ {32,64,128}`, `tile_n=128`, `tile_k ∈ {128,256}` for compile.
3. The `xcd_swizzle` parameter is accepted but unused; wire if perf demands.

## Tests

`op_tests/flydsl_tests/test_fp8_einsum.py`:
- `test_fp8_einsum_compile_trace` — 4 tile shapes, MLIR-level only, runs anywhere.
  All pass.
- `test_fp8_einsum_ue8m0_mode_not_implemented` — guards the stub.
- `test_fp8_einsum_bhr_hdr_bhd_fp32_mode` — end-to-end, `(H=8, D=1024, R=4096)`,
  `B ∈ {4, 32, 128}`. Skips on non-gfx950. **Has not been run yet** — needs you.

Reference impl in the test file is fully self-contained:
- `per_token_cast_to_fp8_x` — matches kernel's online activation quant
- `per_block_cast_to_fp8_y` — per (128,128) block weight quant
- `einsum_fp32_path` — scratch-then-promote per K-128 block (bit-aligned to
  kernel's scratch acc pattern)
- `calc_diff` — DeepGEMM-style `1 - cosine_similarity`; tolerance `< 1e-3`

Larger B (4096, 8192) per design doc are not in the default parametrize set —
they allocate ~256-512 MiB of bf16 X. Add to parametrize if you want them.

## Top three things likely to fail on first hardware run

1. **`ds_bpermute` byte vs dword index**. The helper multiplies the lane index
   by 4 (treating it as bytes). If `flydsl._mlir.dialects.rocdl.ds_bpermute`
   already expects a lane index in dwords, this would be 4× off — and since the
   `s_x` value is from a different lane than intended, results would be
   plausibly-magnitude'd but wrong per-row.

2. **MFMA output row layout assumption** (`src_lane_for_ii`). I assume the 4
   elements of `Vec<4×f32>` returned by `mfma_f32_16x16x32_fp8_fp8` are 4
   successive rows (`lane_div_16*4 + ii`). If instead they're 4 successive
   columns, the per-`ii` `s_x_for_out` remap fetches from the wrong source lane.

3. **`maximumf` NaN propagation**. `arith.maximumf` returns NaN if either input
   is NaN. If inputs ever produce `0 * inf`, swap to `arith.maxnumf` (which
   ignores NaNs).

Less likely but worth checking:
- Whether `bx_h * fx.Index(b_elems_per_head)` overflows i32 for large H*D*R.
  Currently `Index` (i64 on most paths), but the buffer-resource byte budget
  uses i32 indices internally. For `(H=8, D=1024, R=4096)`: `b_elems_per_head =
  4 MiB`, total ≤ 32 MiB — fine. Watch out at much larger H/D/R.
- `acc_init = Vec.filled(4, 0.0, fx.Float32); accs = [acc_init] * N`. Python
  list-multiply of `Vec` creates N references to the same Vec; if Vec is
  immutable (as it appears from the `from_elements` reconstruction pattern in
  the code), this is fine. If it's mutable, each acc needs its own filled().
  All write-paths in `compute_tile_fp32` use `Vec.from_elements(...)` to
  reconstruct, so the references-pointing-at-same-Vec case is safe in practice.

## How to resume

1. Run `pytest op_tests/flydsl_tests/test_fp8_einsum.py -v` on a gfx950 host.
2. If `test_fp8_einsum_bhr_hdr_bhd_fp32_mode` fails: check the three suspects
   above in order. The DPP/ds_bpermute issues will show as per-row-permuted
   wrongness; NaN issues as catastrophic loss.
3. Once fp32 mode is green, tackle Step 4 (UE8M0) with the smoke-kernel approach
   above.
4. Then Step 5: wire `lds_stage=2` and run the perf sweep.

## Quick reference — gotchas the previous agent hit

- `Vec.cast` doesn't exist → use `.to(fx.Float32)`.
- `fx.Int1` doesn't exist → use plain Python int (`0` / `1`) for `word_sel`.
- `cvt_pk_fp8_f32` is positional, not kwarg.
- `buffer_copy_gmem16_dwordx4` idx is in elements (helper does shrui internally
  for elem_bytes=2). Don't pre-multiply.
- `buffer_ops.buffer_load(dtype=fx.Int32)` idx is in dwords, not bytes.
- `arith.maximumf` is the wrapped name (`fx.arith.maximumf`); `arith.absf`
  could not be confirmed wrapped, hence the `max(v, -v)` workaround.
- DPP uses low-level `flydsl._mlir.dialects.rocdl.update_dpp`, not the high
  level `fx.expr.rocdl` import.
