# Round G — fp8 on the matrix core (dequant-in-registers + bf16 WMMA) — page-4 QSA, gfx1250

You are the ROUND-G optimizer (disposable, fresh context). **No GPU**; the root measures. Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-g`
(branch `qsa-index-scorer/opt-g`, base = **champion F commit `f3b37d46`**; kernel sha `7ce0acfc…`).
Never push. Never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`,
`records/` except `records/g2/`) or `records/sealed/`.

## Champion context (measured on a07-3, correctness 22/22 PASS)

Champion F kernel = v0 wrapper + **bf16 path on gfx1250 WMMA 16x16x32** (one workgroup = 4 wave32 = 16 rows ×
64 columns; each wave = 4 query rows × 4 heads = 16-row M tile × four 16-column N tiles; Q in VGPRs; K read
straight from the page-4 cache; ReLU + head sum in-lane; up to 4 request-dedup passes; early -inf exit; all
stores predicated at clamped in-plane offsets, no sentinel) **plus fp8 path = old v0 scalar VALU with
round-B's LDS-staged Q** (1 KiB LDS, one barrier, LDS vector broadcasts instead of global Q re-reads).
Measured (median): decode_bf16_c65536 97.0 µs, prefill512_bf16 216.4, draft_bf16 88.4,
decode_bf16_c32768 77.8; decode_fp8_c65536 **361.5**, draft_fp8 138.0, prefill512_fp8 **2394.0**,
decode_fp8_c32768 206.3. fp8 is still 3.7× slower on decode and **11× slower on prefill512** than bf16 —
that is the single remaining large gap.

## Your round-G task: run the fp8 path on the matrix core too (dequant in registers, bf16 WMMA)

The fp8 cache is E4M3 with a **per-token (per key-row = per column) fp32 scale**. Because
`dot(q, scale_c * k_c) = scale_c * dot(q, k_c)`, you can:
1. Reuse champion F's **bf16 WMMA kernel skeleton** (same M/N tiling, same Q fragment layout, same K
   fragment gather positions, same store/mask logic).
2. Where the bf16 path loads 128 bf16 K values per column, the fp8 path loads 128 **E4M3 bytes** and converts
   each to **bf16 exactly** (E4M3 has 3 mantissa bits so every E4M3 value is exactly representable in bf16;
   use the **same decode semantics the current fp8 kernel uses** — the old scalar decoder is the reference,
   including its treatment of 0x7F/0xFF and signed zero; the CPU contract's exhaustive-256-code check must
   prove bit-exact equality). Convert in registers/lane-local, then feed the same WMMA B fragments.
3. Apply the per-column **fp32 scale after the GEMM**: every output logit of a column is multiplied by the
   same `scale[physical_page, page_off]` (and by `1/sqrt(128)` and any existing scaling exactly as the scalar
   path orders it, so results stay within the frozen tolerance / ideally bit-identical where possible).
   Make sure the scale load is in bounds and its page-table/physical-page path uses the same safety masks.
4. Keep bf16 path byte-for-byte unchanged (do not touch its fragments/stores); add the fp8 path as a separate
   dtype specialisation of the same WMMA builder (e.g. `_build_wmma_kernel(dtype="fp8")` with kernel name
   suffix `_fp8_wmma`), and keep the current LDS-Q scalar fp8 path available as a host-side fallback
   (`QSA_FP8_PATH=scalar` env or equivalent) for shapes you cannot cover.
5. Coverage: all fp8 shapes (any rows/columns/tails/request mix), fp8 scales, page holes → -inf, visible=0,
   invalid token_to_request. Keep the wrapper API/return values unchanged.

## Hard constraints (unchanged)

Stock flydsl **0.3.3.dev886** only (no dev1004 APIs); verified available: `fx.rocdl.WMMA`,
`MmaOpGFX1250_WMMAType`, `fx.make_mma_atom`, `fx.gemm`, `fx.make_tiled_mma`, `fx.make_rmem_tensor`,
`fx.make_tiled_copy_A/B/C`, `fx.SharedAllocator(base_alignment=16, static=True).allocate(...).peek().ptr`,
`fx.gpu.barrier()`, `fx.ptr_load(ptr, result_type=None)`, `fx.ptr_store(value, ptr)`.
No sentinel/OOB stores; clamped predicated stores; i64 descriptor extents; host guards (alignment, dtype,
no silent int64→int32 narrowing); every logit written in [0, visible), -inf elsewhere.

## Verification (CPU only) + deliverable

- Champion F's gates must stay green: `_g2/test_contract_cpu.py` (incl. WMMA lane emulator),
  `_g2/trace_check_cpu.py` 16/16, `_g2/audit_addressing_cpu.py` 26/26 (extend for the fp8 WMMA path),
  `_g2/test_q_lds_cpu.py`; plus a new exhaustive E4M3→bf16 conversion check against the old decoder for all
  256 codes and an fp8-WMMA lane emulator on every fp8 fixture (bit/tolerance comparison vs golden).
- `COMPILE_ONLY=1` trace must `[COMPILED]` all specialisations (bf16, fp8 scalar fallback, bf16_wmma,
  fp8_wmma) — tracing-only, state that.
- Commit on `qsa-index-scorer/opt-g` and report JSON {status, base_commit, commits[], files_sha256,
  what_changed, expected_gain{case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done.
