# ROUND-NATIVE64: int64 query_positions read natively (no per-call clamp kernel)

Base: champion `8a8d368a`. Branch `qsa-index-scorer/opt-native64`.

## Problem (measured a08-1 GPU3, run_perftest graph replay)

Golden fixtures and the ATOM caller pass **int64 `query_positions`**; ROUND-NOSYNC
narrowed them with one device `clamp(..., out=int32)` launch per call. Passing int32
positions instead saves 2.5-6 µs/call (decode bf16 c65536 32.3→28.9 µs, decode bf16
c32768 19.7→17.3, draft fp8 c32768 22.4→16.2).

## Change

* **One load seam, shared by every builder** (`_pos_loaders(pos64)` next to
  `_ld_uniform_i32`): returns the per-lane loader, the SGPR loader, the position
  plane bytes/row and a kernel-name suffix. All five position loads (scalar kernel,
  bf16 `_rows_meta`, `_rows_meta_uniform`, `_r16_stage_active`, fp8 `_rows_meta`)
  and all eight position descriptors go through it. A parallel builder only needs
  `ld_pos, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)`.
* `pos64=False` (int32 callers, **default**): `_ld_i32(rsrc, row * 4)` /
  `_ld_uniform_i32(rsrc, row * 4)`, extent `rows * 4`, unchanged kernel names —
  the same IR as before (ISA identity below).
* `pos64=True` (kernel names `..._qp64`): position plane = zero-copy int32 view
  `[2*rows]` of the caller's int64 tensor, extent `rows * 8`; per-lane path one
  8-byte `buffer_load` per row, SGPR path two dword `s_buffer_load`s (row*8,
  row*8+4). `_sat_pos64(lo, hi)` computes in i32 compare/select only:

  ```
  fits    = hi == (lo < 0 ? -1 : 0)          # v is representable in int32
  result  = fits ? min(lo, INT32_MAX-1)
                 : (hi >= 0 ? INT32_MAX-1 : INT32_MIN)
  ```
* Wrapper (default, non-strict): int64 positions → `pos64=True`, `qp.view(int32)`
  after the unchanged stride rule (`.contiguous().view(-1)` only for stride ≠ 1);
  **no tensor op at all** (census: 0 ops). Builders are called with `**qp_kw`
  (`{}` for int32 → identical lru_cache entry / call as before).
  page_table / token_to_request / context_lens keep the NOSYNC clamp path for
  int64 (rare). `QSA_STRICT_I32=1` unchanged (raising probe, int32 kernel).
  `QSA_PAD_BUCKET=1` (opt-in) narrows positions via `_to_i32_saturated` only on
  the padding route (`_qp_pad_i32`), its kernels are int32-only as before.

## Semantics proof

For an int64 `v` with little-endian words `lo = v mod 2**32` (as signed i32) and
`hi = v >> 32` (arithmetic): `v ∈ [INT32_MIN, INT32_MAX]` ⇔ `hi` equals the sign
extension of `lo`, and then `v == lo`; otherwise `sign(v) == sign(hi)` with
`v ≥ 2**31` iff `hi ≥ 0` (`hi > 0`, or `hi == 0` with `lo < 0`) and `v < -2**31` iff
`hi < 0` (`hi ≤ -2`, or `hi == -1` with `lo ≥ 0`). Hence `_sat_pos64` ≡
`clamp(v, INT32_MIN, INT32_MAX-1)` — **bit-identical** to the ROUND-NOSYNC wrapper
value for every int64, so the whole ROUND-NOSYNC proof (`_to_i32_saturated`
docstring, `ROUND_NOSYNC.md`) carries over unchanged: visible groups exact whenever
the golden `V < 2**29`, every logit mask exact. All seam arithmetic stays inside
i32 (no shifts, no wrap; the trace mock rejects any i32 overflow).

## Verification (no device)

* `test_native64_cpu.py`: real `_ld_pos_i64` / `_ld_uniform_pos_i64` source executed
  by the trace mock over 36 boundary values + neighbourhoods of INT32_MIN/MAX,
  ±2**32, 2**31, int64 extremes + 1024 random → == golden clamp == NOSYNC wrapper;
  offsets (row*8, row*8+4), extent rows*8; visible groups for all 36×36
  (pos, ctx) pairs vs the golden int64 formula. Whole-kernel trace (bf16 WMMA, r16
  and r32, raw K): `pos64=True` on raw int64 boundary positions is **bitwise
  identical** to the int32 kernel on narrowed positions and matches
  `reference_logits`.
* `test_nosync_i32_cpu.py` extended: in-kernel saturation model on the 36
  boundaries, wrapper passes the zero-copy view + pos64, census 0 ops for qp64.
* Structural gates replay the int32 binding of the seam (`native64_seam.py`);
  persist's pinned source hashes are taken after `unseam` (the seam is the only
  change in every pinned function).
* ISA (`host_native64_cpu.sh STAGE isa`): 12 specializations × {base i32, final
  i32, final qp64}; see `native64_isa_evidence.json`.

## Results (a07-3 container, commit 4496f852, kernel sha `1161165c…`)

All 16 CPU gates exit 0 (`native64_cpu_evidence.json`). ISA: every int32
specialization is **byte-identical** to base `8a8d368a` (ISA sha, kernel name,
resources); POS64 variants keep VGPR/SGPR/LDS/scratch unchanged:

| spec | int32 ISA == base | instr i32 → qp64 | VGPR | SGPR |
|---|---|---|---|---|
| bf16 r16 TDM (decode/draft) | yes | 3396 → 3490 (+94) | 97 | 105 |
| bf16 r16 raw | yes | 2024 → 2053 (+29) | 105 | 66 |
| bf16 r32 TDM | yes | 4959 → 5011 (+52) | 183 | 79 |
| bf16 r32 persist8 (prefill) | yes | 34398 → 34427 (+29) | 229 | 86 |
| bf16 r32 raw | yes | 4848 → 4909 (+61) | 153 | 66 |
| bf16 r64 TDM (opt-in) | yes | 9619 → 9761 (+142) | 235 | 103 |
| fp8 r16 (TDM env resolves raw) | yes | 6701 → 6730 (+29) | 248 | 66 |
| fp8 r32 TDM cvtfast | yes | 18551 → 18615 (+64) | 168 | 83 |
| fp8 r32 raw | yes | 19341 → 19404 (+63) | 185 | 69 |
| bf16 scalar | yes | 1884 → 1893 (+9) | 512 | 66 |
| fp8 scalar | yes | 3844 → 3854 (+10) | 204 | 66 |

Per-lane paths: each `buffer_load_b32` of a position becomes one
`buffer_load_b64` plus ~5 VALU (cmp/cndmask/min). bf16 r16 TDM (SGPR path):
`s_buffer_load_b64` ×8 replace 2×`s_buffer_load_b128` (row-adjacent positions no
longer merge) + SALU saturation; the base kernel already spills SGPRs to VGPR
lanes (v79) and qp64 adds 20 `v_readlane_b32` reloads (SGPR 105, at the cap).
These are block-prologue costs, far below the removed 2.5-6 µs clamp launch.
