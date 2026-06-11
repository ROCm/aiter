# Plan: new weight (B) scale preshuffle layout for grouped gfx1250 MoE GEMM

## Goal
Change the **weight (B) scale** e8m0 preshuffle to the layout

```
view(E, remain_n, 2, n_lane, other_k, 2, 2).permute(0, 1, 4, 6, 3, 2, 5)
# memory order: [E, remain_n, other_k, k2_a, n_lane, n2, k2_b]
```

and update the consumer GEMM B-scale read path to match. Activation (A) scales
keep the existing `_grouped_a8w4_preshuffle_e8m0_scale` layout, so the GEMM's
A-scale and B-scale paths stop being symmetric.

## Confirmed decisions
- Scope: **weight (B) scale only** (w1_scale, w2_scale). A-scale untouched.
- **Both producer and consumer** change.
- Memory order: `[E, remain_n, other_k, k2_a, n_lane, n2, k2_b]` (user-confirmed).

## Assumption to lock in step 0
`n_lane = 16` (the WMMA N-lane axis, replacing the current "16"). Then for a raw
weight scale `(E, N, k_scale)` with `k_scale = K//32`:
- `remain_n = N // 32`         (because `remain_n * n2(2) * n_lane(16) = N`)
- `other_k  = k_scale // 4`     (because `other_k * k2_a(2) * k2_b(2) = k_scale`)
- output shape `(E, remain_n, k_scale * 32)`  (cols = `other_k*2*16*2*2`)

Difference vs current B layout `(E, N//4, k_scale*4)`: the new layout moves the
16-lane axis **out of the row dim into the column dim**, so each super-row spans
32 N-rows (vs 4) and the lanes are interleaved with K inside the row. This is the
crux of why the consumer addressing must be rewritten, not just reparameterized.

Index map (raw `(n, ks_idx)` -> output `(super_row, col)`), n_lane=16:
```
n      = remain_n_i*32 + n2_i*16 + lane_i        # remain_n_i, n2_i in{0,1}, lane_i in[0,16)
ks_idx = other_k_i*4 + k2_a_i*2 + k2_b_i         # k2_a_i,k2_b_i in{0,1}
super_row = remain_n_i
col       = ((((other_k_i)*2 + k2_a_i)*16 + lane_i)*2 + n2_i)*2 + k2_b_i
```
(Derive/confirm this exactly in step 1 from the literal view+permute; the kernel
addressing in step 3 is the inverse of this `col` expression.)

## Coupling points (what exists today)
1. `aiter/ops/flydsl/grouped_moe_gfx1250.py`
   - `_grouped_a8w4_preshuffle_e8m0_scale` (shared A+B reference permute)
   - `_grouped_a8w4_prepare_scale_batch` (wrapper; used by test for weight scale)
   - weight scales reshaped at lines ~627-630 (`grouped_w1_scale`, `grouped_w2_scale`)
2. `aiter/ops/flydsl/kernels/moe_scatter_copy_preshuffle_scale.py` — fused producer,
   **activation only** -> NOT touched (weight-only scope).
3. `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py` — consumer:
   - `b_scale_load_rep`, `interleaved_scale_cols_b`
   - `make_desc_bs` (TDM descriptor: tensor_shape/strides/tile_shape/offsets)
   - `_precompute_scale_lane_bases` (shared A/B base formula)
   - `load_scale_b128` / `load_scale_slice_b128`
   - `make_desc_as` / A-scale lane bases stay on the OLD path.
4. `aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py`
   - `_preshuffled_scale_shape` (validation) + scale arg shape checks
5. `op_tests/test_flydsl_grouped_gemm_gfx1250.py` — `_grouped_scale` builds the
   weight scale via `_grouped_a8w4_prepare_scale_batch`.
6. `aiter/ops/flydsl/moe_kernels.py` — references the permute + documents an inverse.

## Hard constraint: arch
This box is **gfx942**. `test_flydsl_grouped_gemm_gfx1250.py` calls
`_require_gfx1250()` and will **skip** the real GEMM here. So:
- Locally verifiable: the torch producer + a pure-torch roundtrip (producer ->
  manual inverse == identity) on gfx942.
- NOT locally verifiable: the actual grouped GEMM numerics. The e2e test must be
  run by the user on a gfx1250 box. Plan delivers a self-contained roundtrip test
  so the index math is proven before touching the kernel.

## Run mechanism on gfx942 (no-real-gemm)
- `--no-real-gemm` (`_mock_grouped_gemm`) patches the grouped GEMM compilers to
  no-ops, so the **consumer kernel never executes** on gfx942 — it is only
  py_compile-checked here and must be validated later on gfx1250.
- What runs under mock: producer `_grouped_scale` + library weight-scale reshape
  (`grouped_moe_gfx1250.py` lines 627-630) + the rest of the tiny-op pipeline.
- `main()` exits early via `is_gfx1250()` (ignores FORCE); patch that guard to
  also accept `AITER_FORCE_GFX1250=1` so the no-real-gemm run works on gfx942.
- Success signal here = the pipeline executes with the new shape (no shape/throw);
  numerics are NOT validated under mock (use `--scenario bench`, which has no
  rel_l2 assert, or accept that verify's rel_l2 is meaningless under no-op GEMM).

## Quality gates (per user)
- Write ALL code (producer + consumer + shape checks + test wiring).
- Review the diff 3 times against the index math before running.
- `python -m py_compile` every changed file.
- Change the weight-scale shuffle in the test and run `--no-real-gemm` on gfx942.

## Implementation steps

### Step 0 — confirm n_lane and exact index math (read-only)
Write the literal `view(...).permute(0,1,4,6,3,2,5).reshape(...)` in a scratch
numpy/torch snippet, dump the `(n, ks_idx) -> flat` map for a tiny case
(E=1, N=64, k_scale=8), and confirm it equals the closed-form `col` above. Lock
`n_lane=16`. (Gate for everything else.)

### Step 1 — producer: new B-scale preshuffle (torch reference)
In `grouped_moe_gfx1250.py`, add `_grouped_b_scale_preshuffle_e8m0` (new layout)
and route weight-scale prep through it:
- New fn: `view(E, N//32, 2, 16, k_scale//4, 2, 2).permute(0,1,4,6,3,2,5).reshape(E, N//32, k_scale*32)`.
- Add `_grouped_b_scale_prepare_batch` mirroring `_grouped_a8w4_prepare_scale_batch`
  but with the new shape, divisibility checks (N%32, k_scale%4), and
  preshuffled-shape passthrough.
- Update `grouped_w1_scale` / `grouped_w2_scale` (lines ~627-630) to the new fn.
- Keep A-scale (`grouped_a1_scale`, `grouped_a2_scale`) on the old fn.

### Step 2 — local roundtrip test (runs on gfx942)
Add `op_tests/test_grouped_b_scale_layout.py` (pure torch, no kernel):
- random `(E, N, k_scale)` u8 -> new producer -> manual inverse -> assert equal.
- assert output shape `(E, N//32, k_scale*32)`.
- a few shapes incl. the test's real ones (rows=2*inter and rows=K).
Run: `AITER_USE_SYSTEM_TRITON=1 python -m pytest op_tests/test_grouped_b_scale_layout.py`.
This is the correctness gate for the math before kernel edits.

### Step 3 — consumer GEMM B-scale read path
In `gemm_mxscale_gfx1250.py`, split B-scale from the shared scale path:
- Recompute B-scale geometry from the new layout: new `b_super_rows = N//32`,
  new per-tile interleaved cols, new `b_scale_load_rep` semantics.
- Rewrite `make_desc_bs` tensor_shape/strides/tile_shape/global_offset for
  `(E, N//32, k_scale*32)`.
- Add a B-specific lane-base + load helper that inverts the new `col` formula
  (lane_i from `lane16`, n2_i from `lane_kgrp` or warp index, k2_a/k2_b from the
  ks loop), replacing `_precompute_scale_lane_bases`/`load_scale_b128` for B only.
- Leave `make_desc_as` and A-scale addressing unchanged.
- Update `_emit_wmma` B-scale operand indexing to the new fragment order.

### Step 4 — validation shape helpers + arg checks
- `moe_grouped_gemm_mxscale_gfx1250.py`: add `_preshuffled_b_scale_shape`
  (returns `(N//32, k_scale*32)`) and use it for B-scale (`scale_w`) checks;
  keep `_preshuffled_scale_shape` for A.

### Step 5 — test wiring
- `test_flydsl_grouped_gemm_gfx1250.py`: point `_grouped_scale` (weight path,
  lines 354-355 / 484-485) at the new B-scale prepare fn. Activation scale prep
  stays as-is.

### Step 6 — verify
- Local (gfx942): step-2 roundtrip + `_grouped_scale` shape sanity.
- gfx1250 (user-run): `AITER_USE_SYSTEM_TRITON=1 python -m pytest -q \
  op_tests/test_flydsl_grouped_gemm_gfx1250.py` (a4w4 + a8w4 correctness cases),
  then the `--scenario verify` direct run. Report rel_l2 vs the existing tols
  (VERIFY_TOL_A4W4 / A8W4 = 0.02).

## Risks / notes
- A/B asymmetry: the GEMM currently shares scale addressing; splitting B out is
  the largest edit and the main regression risk. Keep A path byte-identical.
- `n_lane != 16` would invalidate `remain_n`/shape math — step 0 must confirm.
- Fused activation scatter-copy kernel is intentionally out of scope (A only).
- Can't prove GEMM numerics locally; step-2 roundtrip is the local proxy.

## Rollback
Single-format flag or revert the weight-scale prep routing (steps 1/5) to the old
shared fn; consumer step-3 changes are isolated to the B path.
