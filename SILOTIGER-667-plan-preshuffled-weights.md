# SILOTIGER-667 — Opt-in `k_contiguous` / `preshuffled` weights

Let warp-decode MoE **read the same preshuffled weight buffer** as the MFMA MoE
kernels (`mxmoe_gemm_v2` / a16w-mix `make_preshuffle_b_layout`), as an **opt-in**
layout. Users pick **`k_contiguous`** (today’s staged layout, packed along K) or
**`preshuffled`**. Default stays **`k_contiguous`**. Compute stays packed
`v_dot2_f32_bf16` (cvt + G7 drain). This is a second addressing path, not a
matrix-core rewrite.

**In scope:** `aiter/ops/flydsl/kernels/warp_decode_moe.py`,
`aiter/ops/flydsl/warp_decode_moe.py`, and tests as a gate (`k_contiguous`
cases stay; add `preshuffled` cases). Host API grows an explicit layout
selector.

**Out of scope (this track):** `fx.gemm` / MFMA / LDS/`SharedAllocator`;
unpacked e4m3/e2m1 global layouts; requiring activations to be preshuffled;
retiring the `k_contiguous` path; finishing modernization subtask 5 leftovers
(K-loop packed activations, Phase 1 primitives, FP4 i8 E8M0 on `buffer_ops`
unless a later decision shares **scale** layout too). Scale folding / block2d
into convert remains a separate TODO.

Work the subtasks **in order**. Later items assume earlier ones have landed.
Each subtask is surgical: verify correctness (and a spot-check of median
kernel time on the path that changed) before starting the next.

## Progress

Track here as work lands (leave unchecked until that item is done). Per-subtask
gates: op_test with `FLYDSL_RUNTIME_ENABLE_CACHE=0` on GPU 1; plus a G9/667
spot-check when the hot loop or wait/reduce path changed.

- [x] 1. Host API: opt-in `k_contiguous` | `preshuffled` (default `k_contiguous`)
- [x] 2. Pin the preshuffle contract per dtype (same helper fused MoE uses)
- [x] 3. i32/byte kpack views + i64 expert base on the `preshuffled` path
- [ ] 4. Lane→kpack gather map (wave still owns output scalars; `v_dot2`)
- [ ] 5. Op_test both layouts; fail loudly on illegal `preshuffled` shapes

## Locked decisions

These locks apply to **this track**.

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 1**
  (`HIP_VISIBLE_DEVICES=1`).
- **Surgical, behavior-preserving (`k_contiguous` default).** The existing
  `k_contiguous` path keeps today’s numerics (cos ≥ 0.999 vs cases already in
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`). Do not change
  kVector defaults, `dot2_acc`, `kh_per_warp`, split-K policy, or scale folding
  as a side effect of adding `preshuffled` B. `preshuffled` is a
  **compile-time** second path: same math on a different weight address map.
  Do not silently require `preshuffled` weights; do not change default
  dispatch.
- **Keep ISA-level helpers.** `v_dot2_f32_bf16` inline asm, `s_nop 2` / G7
  drain, `cvt_scalef32_pk_bf16_{fp8,fp4}`, and E8M0 `shl 23` stay; there is no
  `fx.gemm` atom for `v_dot2`. Small-M padding is still why this kernel avoids
  matrix cores.
- **Compile-time specialize layout.** `const_expr` / two builders — not a
  runtime branch inside the K loop. Do not introduce SSA live-outs from
  `const_expr` if/else (same frontend restriction as modernization subtask 4).
- **Packed dwords, not element grids.** Layout of the **pack** is i32 (or byte)
  kpack modes (`klane`, `nlane`, K-tile, kpack, …), then `make_buffer_tensor` /
  `fx.copy` of dword tiles — the same contract as tiled-MMA B, not a
  `Float8`/`Float4` element grid.
- **i64 expert base still applies.** Large-E pools (`E*I*H` bytes ≥ 2³¹) still
  fold the expert base into the buffer descriptor; then view **in-expert**
  kpack tiles. In-expert dword offsets stay i32-safe. Do not invent unpacked
  e4m3/e2m1 grids to “fix” addressing.
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale HSACO
  cannot mask a bad rewrite.
- **Gate.**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py -v
  ```
  Spot-check one FP8 and one MXFP4 decode shape on the existing 667 bench after
  any change that touches the hot loop or wait/reduce path (scheduler-sensitive).
  `k_contiguous` rows must stay within noise of HEAD; `preshuffled` rows are a
  new path (correctness first; bandwidth may be worse than a `k_contiguous`
  load).

## Subtasks

### 1. Host API: opt-in `k_contiguous` | `preshuffled` (default `k_contiguous`)

Wrappers (`flydsl_warp_decode_moe` and the staged kernels) today document
`k_contiguous` layouts only (`w_gate/w_up` `[E, INTER, HIDDEN]`, `w_down`
`[E, HIDDEN, INTER]`, packed along K). Add an explicit selector, e.g.
`weight_layout="k_contiguous"|"preshuffled"`, default `"k_contiguous"`.

- [x] Default dispatch and existing call sites stay `k_contiguous`; docstring
      states both contracts.
- [x] Wrong layout for the buffer is a hard error (do not guess from strides).
- [x] **Done when:** `k_contiguous` launches unchanged; `"preshuffled"` is
      unreachable until subtasks 3–4 land (flag may reject `"preshuffled"`
      until then).

### 2. Pin the preshuffle contract per dtype

FP8, MXFP4, and BF16 are **not** one permutation. Pin the same helper fused
MoE already uses (`make_preshuffle_b_layout` and/or the a16w-mix /
`mxmoe_gemm_v2` views): N-major vs K-major, `kpack_bytes` 8 vs 16,
`elem_bytes` 1 vs 2.

Preshuffle is built around N/16 and K packed into 16B (or 8B) kpacks.
INTER/HIDDEN that are legal for `k_contiguous` warp-decode may be **illegal**
for `preshuffled` B — reject those shapes on the `preshuffled` path.

Sharing fused-MoE **weights** does not automatically share **scale** buffers
(block2d / E8M0 vs MFMA group scales). This track consumes `preshuffled` **B**
only unless a later decision says otherwise.

Pinned on CDNA (`aiter.ops.shuffle.shuffle_weight` / `make_preshuffle_b_layout`
N-major, permute `(0,1,3,4,2,5)`). Not gfx1250 `shuffle_weight_gfx1250`.
Warp-decode weights are **split** (`w_gate`, `w_up`, `w_down`), so MXFP4 uses
the fused **w2** helper (`gate_up=False`), not fused stage1
`shuffle_weight_a16w4(w1, 16, True)` on `[E, 2*INTER, K]`.

| dtype | host helper | device layout | kpack_B | elem | N (gate / down) | packed K (last dim) |
| --- | --- | --- | --- | --- | --- | --- |
| FP8 e4m3 | `shuffle_weight(w, layout=(16, 16))` | `make_preshuffle_b_layout(..., kpack_bytes=16, elem_bytes=1, k_major=False)` | 16 | 1 | INTER / HIDDEN `% 16 == 0` | HIDDEN / INTER `% 64 == 0` |
| BF16 | `shuffle_weight(w, layout=(16, 16))` | `make_preshuffle_b_layout(..., kpack_bytes=16, elem_bytes=2, k_major=False)` | 16 | 2 | INTER / HIDDEN `% 16 == 0` | HIDDEN / INTER `% 32 == 0` |
| MXFP4 | `shuffle_weight_a16w4(w, 16, False)` | a16wmix `layout_b` `(N/16, (K/2)/64, 4, 16, 16)` | 16 | 1 (bytes) | INTER / HIDDEN `% 16 == 0` | `HIDDEN//2` / `INTER//2` `% 64 == 0` (unpacked K `% 128`) |

Host wrappers reject any other N/K on `weight_layout='preshuffled'` (do not
guess from strides). Legal `preshuffled` shapes still raise *not implemented*
until subtasks 3–4.

- [x] Document the permutation + kpack size per weight dtype (gate/up vs down
      N/K: INTER vs HIDDEN).
- [x] List shape constraints (N % 16, K packing) and reject the rest.
- [x] **Done when:** a short table in this file (or a kernel comment) names the
      exact layout helper per dtype; no ad-hoc second permutation.

### 3. i32/byte kpack views + i64 expert base on the `preshuffled` path

`fx.make_view` + `fx.make_layout` over kpack modes, then
`fx.rocdl.make_buffer_tensor` / `fx.copy` of dword tiles. Fold
`e * bytes_per_expert` into the descriptor (`_ptr_rsrc_off` or i64-base
`make_buffer_tensor` equivalent); then view in-expert kpack tiles.

The `k_contiguous` path may keep `load_i32_words` + `buffer_ops` until a later
modernization item moves it. Do not block `preshuffled` B on that.

- [x] `preshuffled` weight loads go through kpack buffer-resource views.
- [x] Large-E (i64 expert base) works on `preshuffled` B the same way as
      `k_contiguous`.
- [x] **Done when:** no fake unpacked FP8/FP4 element layout; expert base is
      folded, in-expert offsets i32-safe.

Host still rejects `weight_layout='preshuffled'` until subtask 4 (lane map +
numerics vs fused-MoE buffers). Builders take compile-time `preshuffled=`;
the kpack path is covered by the in-expert load probe. Do **not** `if
preshuffled` in the `@flyc.kernel` body (SSA live-out / `w_word_base`
`NameError`); dispatch in Python helpers that return a complete bundle.

### 4. Lane→kpack gather map (still `v_dot2`)

Wave 64 still owns one (or `kh_per_warp`) output scalar(s). Each lane’s
K-chunk is gathered from preshuffle slots instead of a `k_contiguous` row
(`w_row * (H//4) + k_base//4`). Activations stay `k_contiguous` (packed i32
dwords in K).

Specialize at compile time (`const_expr(preshuffled)` / two builders). Flatten
address math so nothing is defined only inside a `const_expr` if/else and
used after the branch.

- [ ] New lane→kpack map for gate/up and down.
- [ ] Compute stays `v_dot2` (`cvt_scalef32_pk_bf16_{fp8,fp4}` + G7 drain).
- [ ] **Done when:** `preshuffled` B matches today’s numerics on the
      preshuffled buffer the MFMA path already uses (no extra unpack staging);
      `k_contiguous` path still matches existing op_test.

### 5. Op_test both layouts; fail loudly on illegal `preshuffled` shapes

Feed **already-preshuffled** tensors (reuse fused-MoE `shuffle_weight`). Do
not preshuffle inside the kernel. Keep `k_contiguous` cases until that path
is retired (not this track).

- [ ] Op_test covers `k_contiguous` (existing) and `preshuffled` (new) for the
      dtypes that have a pinned contract.
- [ ] Illegal `preshuffled` shapes raise; default `"k_contiguous"` still runs
      today’s decode shapes.
- [ ] One FP8 and one MXFP4 decode bench spot-check (`k_contiguous` within
      noise; `preshuffled` recorded, not a merge-gate vs `k_contiguous` µs).
- [ ] **Done when:** full
      `test_flydsl_warp_decode_moe.py` pass on GPU 1 with cache off; both
      layouts covered where the contract exists.

## Non-goals (do not pull into this plan)

- Switching compute to `fx.gemm` / scaled MFMA.
- Preshuffling activations.
- Unpacked e4m3/e2m1 global layouts.
- Folding FP8 block2d scale into the convert (separate TODO).
- Finishing `SILOTIGER-667-plan-modernize.md` subtask 5 leftovers as a
  prerequisite.
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=1`.
- Treating `preshuffled` vs `k_contiguous` kernel time as a correctness gate
  (gather can be slower than a `k_contiguous` load).
