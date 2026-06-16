# Plan v3: weight (B) scale preshuffle — N4K8 layout (gfx1250 grouped MoE GEMM)

> **Supersedes** both `plan/weight_bscale_layout_change.md` (作废/obsolete) and
> `plan/weight_bscale_layout_v2.md` (its permute differs from the one locked here).
> The 4 lane32 commits (`8e1c9c7d2`, `8c499ac31`, `face6b9c3`, `3614aa181`) may be
> fully rolled back. We do **not** git-revert them; we **rewrite the byte math
> in place** and reuse their plumbing (the `b_scale_layout` flag, validation
> helper, descriptor scaffolding, test wiring). End state == rollback + reimplement.

## Goal (locked layout, user-given)
```python
scale.view(E, remain_n, 4, 16, remain_b, 2, 4).permute(0, 1, 4, 2, 5, 3, 6)
#   remain_n = N // 64,   remain_b = k_scale // 8   (k_scale = K // 32)
#   memory order: [E, remain_n, remain_b, p(4), q(2), lane(16), r(4)]
```
Replace the weight (B) e8m0 scale preshuffle with this view/permute/reshape and
update the consumer GEMM B-scale read path. Activation (A) scales are untouched.

Reference for the WMMA scale contract / op_sel semantics:
`/workspace/ffm/FlyDSL/tests/kernels/test_wmma_scale_sample.py`.

## Why this layout (vs lane32) — the key conceptual fix
`V_WMMA_SCALE_F32_16X16X128` consumes a **full i32 per lane = 4 e8m0 bytes = the
4 K-blocks of one WMMA-K=128 step** (the reference proves 1·1 + 0.5·0.5 + 0.25·
0.25 + 0.125·0.125 over the 4 blocks → 42.5). `op_sel` (`scaleAType`/`scaleBType`)
selects **which physical lane-half** supplies the scale (op_sel=0 → lanes 0-15,
op_sel=1 → lanes 16-31), **not** which bytes inside the i32.

lane32 instead split the i32 into 2-block half-planes keyed by `op_sel` and
needed a 2-plane gather. The new layout puts the 4 K-blocks `r=0..3` **contiguous**
(innermost), so each lane reads its complete `scaleB` i32 with **one
`ds_load_b32`** and `op_sel` cleanly = the N-tile lane-half. No gather, no parity
filler, no K-split-on-op_sel.

## Index math (verified in numpy — `col` ↔ permute and consumer read both pass)
Raw byte at `(n, ks)`:
```
n  = super*64 + p*16 + lane     # super = n//64, p = (n//16)%4, lane = n%16
ks = remain_b*8 + q*4 + r       # remain_b = ks//8, q = (ks//4)%2, r = ks%4
```
Output (per expert) shape `(N//64, k_scale*64)`; row = `super`, column:
```
col = remain_b*512 + p*128 + q*64 + lane*4 + r
```
WMMA-K step `s = remain_b*2 + q` (s ∈ [0, k_scale//4); 4 e8m0 blocks per step).
The 4 contiguous bytes `r=0..3` at `dword_col = remain_b*512 + p*128 + q*64 + lane*4`
are exactly the `scaleB` i32 for N-row `n = super*64 + p*16 + lane`, step `s`.

> **NOTE the stride correction vs first reading:** `p` stride is **128**, not 256
> (`p` is followed by q·lane·r = 2·16·4 = 128). Confirmed bijective in numpy.

## Consumer lane → (p, op_sel) mapping (grouped cfg: tile_n=64, n_warp=2)
- `warp_tile_n = tile_n//n_warp = 32` ⇒ `wmma_n_rep = warp_tile_n//WMMA_N_EFF = 32//16 = 2`.
- Thread layout (existing): `lane_kgrp = (tid//16)%2`, `lane16 = tid%16`,
  `wave_n_idx ∈ {0,1}`.
- A warp owns 32 N-rows = 2 N-tiles of the 64-row super-row:
  `p = wave_n_idx*2 + lane_kgrp`  (wave_n_idx=0 → p∈{0,1}; wave_n_idx=1 → p∈{2,3}).
- With `use_scale_opsel`: `wn=0 → op_sel=0 (lanes 0-15)`, `wn=1 → op_sel=1 (lanes 16-31)`,
  `b_scale_idx = wn//2 = 0`. So **one loaded dword per warp per step**, shared by
  both `wn` via op_sel; the per-lane address already encodes `p` through
  `lane_kgrp` (and `wave_n_idx`). This is the case the reference validates.

Per-lane LDS byte base (super-local 0, s=0, idx=0), `ROW_BYTES = scale_k_per_tile*64`:
```
base = super_local*ROW_BYTES + wave_n_idx*256 + lane_kgrp*128 + lane16*4
addr(s, idx) = base + idx*256 + (s//2)*512 + (s%2)*64
```
(`wave_n_idx*256` = 2 N-tiles × p-stride 128; `idx` only nonzero if a config ever
has `wmma_n_rep > 2`.)

## Step 0 — RESULTS (verified on gfx942 via numpy + code inspection)
**All locked. The values below supersede the earlier assumptions in this doc.**

- **Grouped runtime config** (`grouped_moe_gfx1250.py:447,481-482`): `tile_m=64`,
  `tile_n = n_warp*64`, `tile_k=256`, `m_warp=1`, `n_warp∈{2,4}`. Hence
  **`warp_tile_n = tile_n//n_warp = 64` always**, `scale_k_per_tile = 256//32 = 8`
  (= exactly one `remain_b` block ⇒ n4k8 columns are contiguous per k-tile),
  `k_wmma_steps = 256//128 = 2` (ks=0,1 → q=0,1).
- **`use_scale_opsel = False` always** for the grouped path. ⇒ **n4k8 enforces
  `use_scale_opsel == False`** (assert in flag parse). `b_opsel=0`, `b_scale_idx=wn`.
- **a8w4 ⇒ ROW_MAJOR_STREAMING** (`gemm_mxscale:459` gates COL_BAND on `is_fp4`):
  `wmma_n_rep = 64//16 = 4`, warp owns a full super-row, **p = wn**. Consumer:
  `_load_b_and_scales`→`_emit_wmma`, `b_scales[wn]`.
- **fp4 ⇒ FP4_COL_BAND**: `wmma_n_rep = 64//32 = 2`, each fp4-wn = 32 rows = 2
  N-tiles via the 32 lanes, `_bank_half_wn=1`, `_bank_half_b_scale_rep=2`.
  Consumer: `_load_b_half_bundle`→`_emit_group_rows`, `b_scales[wn_local*2]`,
  `scaleAType=0`.
- **Verified consumer addressing (numpy, both formats pass):** `LDS[super_local
  row][col] = gmem[expert, super0+super_local, rb*512 + col]`. Per-lane `ds_load_b32`:
  ```
  ROW_BYTES = scale_k_per_tile*64 = 512    super_local = warp_n_base//64 = wave_n_idx
  a8w4:  base = super_local*512 + lane16*4                  off = wn*128   + ks*64
  fp4 :  base = super_local*512 + lane16*4 + lane_kgrp*128  off = fpwn*256 + ks*64
  ```
  Each `ds_load_b32` returns the full 4-K-block scaleB i32 (r=0..3) for one
  (N-row, WMMA-K step). a8w4: op_sel=0 → only lanes 0-15, no lane_kgrp, p=wn.
  fp4: all 32 lanes, p = 2*fpwn + lane_kgrp.
- **Descriptor:** lane32 `*32/÷32` → `*64/÷64`; fold=64, inner_off=`k_scale_off*64`,
  tile `(tile_n//64, scale_k_per_tile*64)`, outer `blk_n//64`, tensor
  `(batch*(N//64), k_scale*64)`, batch_bs_base `B_TOTAL_N//64`.
- **ds-load counts:** full (a8w4) = `wmma_n_rep`; half (fp4) = `wmma_n_rep//2`.
- **⚠ TEST CAVEAT (critical):** the test sets ALL weight scales to byte 127 = 1.0,
  so GEMM output is **insensitive to B-scale layout** — any non-crashing layout
  passes (why lane32 "passed but correctness unknown"). **Validating n4k8 on
  gfx1250 requires VARIED weight scales.** Step 2 roundtrip proves the
  producer↔consumer-address math; the fp4 32x16 lane mapping + e2e numerics still
  need a varied-scale gfx1250 run.

## Step 0 — original notes (now resolved above)
The producer math and the `use_scale_opsel=True` 16x16 read are **verified**.
Three items still to pin before/while editing the consumer:

0a. **`use_scale_opsel` actual value** for the grouped a8w4/fp4 configs in
   `moe_grouped_gemm_mxscale_gfx1250.py` (defaults shown are `False`, tuned cfgs
   may differ). If `False`: `b_opsel=0` always, `b_scale_idx=wn`, every `wn`
   reads lanes 0-15 → the 2 N-tiles can't both live in lanes 0-15 with op_sel.
   Decide: mirror lane32's `[::2]` handling and confirm the non-opsel path's
   lane→row contract, or assert the grouped path always sets `use_scale_opsel`.
   **Inspect the real cfg and gate.**

0b. **fp4 (32x16x128, A/B-swapped) b-scale.** fp4 emit uses `b_scales[wn*2]` with
   `scaleAType=0` fixed (no op_sel on the weight scale) and `b_scale_load_rep =
   warp_tile_n//WMMA_M`. One 32x16 WMMA covers **32 weight-N-rows in 32 lanes**,
   so the scale i32 for 32 rows spans all 32 lanes (no lane-half split). Derive
   the fp4 per-lane address: lane `L∈[0,32)` → N-row `super*64 + p*16 + (L%16)` in
   tile `p = wave_n_idx*2 + L//16`, still `addr = …lane_kgrp*128 + lane16*4…`, but
   indexed `b_scales[wn*2]` not `wn//2`. Confirm in numpy with a 32-lane sweep.
   **Gate.**

0c. **op_sel half ↔ data half** on real hardware. The mapping
   `op_sel=lane_kgrp` is only *provable* on gfx1250. Locally we assert it matches
   the reference's op_sel semantics; user validates numerics on gfx1250.

## Touch points (files) — in-place rewrite, rename `lane32`→`n4k8`
1. `aiter/ops/flydsl/grouped_moe_gfx1250.py`
   - `_grouped_b_scale_preshuffle_e8m0` (~213-225): new view/permute/reshape,
     checks `N%64==0`, `k_scale%8==0`; update the block comment (~192-211).
   - `_grouped_b_scale_prepare_batch` (~228-251): `preshuffled_shape` →
     `(experts, rows//64, k_scale*64)`.
2. `aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py`
   - `_compile_base_a8w4_gemm` (~729): `b_scale_layout="n4k8"`.
   - any `_preshuffled_b_scale_shape`/arg-shape check → `(rows//64, k_scale*64)`,
     `rows%64`, `k_scale%8` (add if present; v2 referenced lines ~281-292).
3. `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py` (consumer)
   - Flag parse (~119-123): accept `"n4k8"`, set `b_n4k8` (rename `b_lane32`).
   - ds-load counts `_b_scale_ds_loads_full/_half` (~297-299): **one `ds_load_b32`
     per N-tile-pair per step**. full = `(wmma_n_rep+1)//2` (idx count); half = 1.
     Recompute `_bs_ds_loads`/`_b_half_scale_loads` so `s_wait_dscnt` matches.
   - `batch_bs_base` (~554-560): `B_TOTAL_N // 64`.
   - `make_desc_bs` n4k8 branch (~684-708): tensor_shape `(batch*(N//64),
     K_scale*64)`, strides `(K_scale*64, 1)`, tile_shape `(tile_n//64,
     scale_k_per_tile*64)`, `outer_off = blk_n//64`, `inner_off = k_scale_off*64`.
   - `_precompute_b_scale_n4k8_base` + `_load_b_scale_n4k8` (~948-975):
     per-lane base `wave_n_idx*256 + lane_kgrp*128 + lane16*4`; load
     `(wmma_n_rep+1)//2` dwords at `idx*256 + (s//2)*512 + (s%2)*64`; **single
     `ds_load_b32` each**, return them indexed so `b_scales[wn//2]`(+op_sel) /
     `b_scales[wn*2]`(fp4) select correctly. No parity filler.
   - `_load_b_and_scales` (~985-992): call `_load_b_scale_n4k8`; keep the
     `use_scale_opsel` `[::2]` semantics consistent with the new return shape.
   - `_load_b_half_bundle` a8w4 (~1265-1281): call the new loader for the bank-half.
   - `_emit_wmma` (~1021-1052): mapping unchanged in shape — keep `b_scales[wn//2]`
     + `b_opsel=wn%2` (16x16) and `b_scales[wn*2]` + `scaleAType=0` (fp4); the new
     loader returns the matching index order.
   - ds-read hint counts (~1420-1447): update for the new full count.
4. `op_tests/test_flydsl_grouped_gemm_gfx1250.py`: `_grouped_scale` routes through
   `_grouped_b_scale_prepare_batch` — picks up the new shape automatically; verify
   no hardcoded `//32`/`*32` for the weight scale remains.

## New constraints
`N % 64 == 0` and `k_scale % 8 == 0` ⇒ `K % 256 == 0` (stricter than lane32's
N%32, K%128). Validate the test's real dims; raise a clear error otherwise.
Also `tile_n % 64 == 0` (descriptor row dim is in 64-row super-rows).

## Implementation steps
1. **Producer** (`grouped_moe_gfx1250.py`): new preshuffle body + checks + comment.
   ```python
   remain_n = N // 64; remain_b = k_scale // 8
   g = scale.view(E, remain_n, 4, 16, remain_b, 2, 4)
   g = g.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
   return g.reshape(E, remain_n, k_scale * 64)
   ```
2. **Local roundtrip test** (`op_tests/test_grouped_b_scale_layout.py`, pure torch,
   runs on gfx942): random `(E,N,k_scale)` u8 → producer → closed-form inverse →
   assert identity; assert shape `(E, N//64, k_scale*64)`; cover the test's real
   dims (rows = 2*inter, rows = K). This is the local correctness gate.
3. **Validation shape** (`moe_grouped_gemm_mxscale_gfx1250.py`): new shape/checks;
   flip `b_scale_layout="n4k8"`.
4. **Consumer** (`gemm_mxscale_gfx1250.py`): rewrite the `b_n4k8` branch per
   touch-points (3); keep the A path byte-identical.
5. **Three reviews (no execution):**
   1. producer permute ↔ closed-form `col`; output shape; divisibility.
   2. consumer `addr` ↔ `col`; `p = wave_n_idx*2 + lane_kgrp` ↔ weight N-tile
      order (`_precompute_b_lane_bases`/`load_b_frag`); op_sel half ↔ data half;
      descriptor strides/tile/offsets; fp4 (Step 0b) vs a8w4/fp8 vs use_scale_opsel.
   3. ds-load **counts** vs actual loads (`s_wait_dscnt`/`_bs_ds_loads`/half);
      A path untouched; all call sites updated; rename complete (no `lane32` left).
6. **Local verify + handoff:** `python -m py_compile` every edited file; run the
   Step-2 roundtrip (`AITER_USE_SYSTEM_TRITON=1 python -m pytest -q
   op_tests/test_grouped_b_scale_layout.py`). Hand off to user for gfx1250:
   `AITER_USE_SYSTEM_TRITON=1 python -m pytest -q
   op_tests/test_flydsl_grouped_gemm_gfx1250.py` (a4w4 + a8w4), report rel_l2 vs
   VERIFY_TOL (0.02).

## Arch constraint
This box is **gfx942**; the grouped GEMM consumer is **gfx1250-only**. Locally
provable: producer↔inverse roundtrip (numpy/torch) + `py_compile`. GEMM numerics
MUST be validated by the user on gfx1250.

## Risks
- **op_sel/p split (Step 0a/0b/0c)** — the only pieces not fully provable locally;
  wrong split = silently wrong numerics on gfx1250.
- ds-load **count** mismatch desyncs `s_wait_dscnt` → races. Recompute carefully.
- Divisibility `K%256`/`N%64`/`tile_n%64`: raise early, don't truncate.
- fp4 (32x16 swap, `scaleAType=0`) vs a8w4/fp8 (16x16 op_sel) differ — handle both.

## Rollback
Set `b_scale_layout="interleaved"` (consumer changes isolated to the `b_n4k8`
branch) and revert the producer preshuffle to the shared interleaved fn.
