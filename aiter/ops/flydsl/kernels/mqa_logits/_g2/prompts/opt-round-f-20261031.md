# Round F — merge champion-B fp8 LDS-Q staging into champion-C (WMMA bf16) kernel

You are the ROUND-F optimizer (disposable, fresh context, NO GPU). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-f`
(branch `qsa-index-scorer/opt-f`, base = **champion C commit `177419ad`**; kernel sha `90e9de97…`). Never push;
never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/` except
`records/g2/`) or `records/sealed/`.

## Situation / goal

- Champion C (`177419ad`) made **bf16** fast with a gfx1250 **WMMA 16x16x32** path (kernel builder
  `_build_wmma_kernel`, bf16 only) and left **fp8** on the old v0 scalar builder (`_build_kernel`) —
  so fp8 is correct but ~15 % slower than champion B (fp8 decode 413.8 µs vs B's 360.6 µs).
- Champion B (`c8cbaeb0`, in worktree `..\qsa-v0` on branch `qsa-index-scorer/opt-b2`) had found a big fp8
  win with a tiny **LDS-staged Q**: each block stages the row's Q (4 heads × 128 bf16 = 1 KB = 256 packed
  i32) into aligned LDS with one scalar in-bounds load + disjoint LDS store per lane, one unconditional
  `fx.gpu.barrier()`, then the inner loop replaces the 64 global Q vector loads with `fx.ptr_load(ptr,
  result_type=Vec(4,Int32))` LDS broadcasts (same bf16 bit unpacking). That change is in
  `git -C ..\qsa-v0 diff 1644eeb5 c8cbaeb0 -- aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py`.

## Your task

**Port champion B's LDS-Q staging into champion C's fp8 build path only.** Keep C's bf16 WMMA path
byte-for-byte unchanged (it is the champion's big win; do not touch its code, only the fp8 builder may
change). Port the LDS staging edits by hand where needed (the files diverged because C restructured builders);
do not paste C's bf16 edits into fp8 or vice versa. fp8 arithmetic order, E4M3 software decode, fp32 scale
application, masks, descriptors, host guards and wrapper API must stay bit-identical to C's current fp8 path
(same results as today on every fp8 case).

Constraints unchanged: stock flydsl **0.3.3.dev886** only (no dev1004 APIs); one `SharedAllocator` per
kernel; no sentinel/OOB stores; clamped predicated stores; i64 descriptor extents; host guards.
Verified LDS API: `fx.SharedAllocator(base_alignment=16, static=True).allocate(struct_or_int,
alignment=None).peek().ptr`, `fx.gpu.barrier()`, `fx.ptr_load(ptr, result_type=None)`,
`fx.ptr_store(value, ptr)`, `fx.memref_load_vec/store_vec`.

## Verification (CPU only; keep all champion-C gates green)

- `python3 _g2/test_contract_cpu.py` (C's extended version incl. the WMMA lane emulator) — PASS.
- `python3 _g2/trace_check_cpu.py` — PASS 16/16 (C's symbolic run of the real kernel body).
- `python3 _g2/audit_addressing_cpu.py` — PASS; extend with the fp8 LDS-Q accesses (LDS in-bounds) if the
  audit models them.
- `COMPILE_ONLY=1` trace (`_g2/compile_only.py`, tracing-only, not a real build): bf16, fp8 and bf16_wmma
  specialisations must all `[COMPILED]`.
- Also add a check that the fp8 path now performs **zero global Q vector loads in the inner loop** (LDS reads
  instead), and that the LDS staging is one writer per dword + aligned consumers (B's `_g2/test_q_lds_cpu.py`
  in `..\qsa-v0` already does this — adapt, don't break it).

## Deliverable

Commit on `qsa-index-scorer/opt-f` (never push) and report JSON {status, base_commit, commits[],
files_sha256, what_changed, expected_gain{case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}.
Expected: fp8 cases recover to champion-B levels (decode_fp8 ~360 µs) while bf16 keeps champion-C levels.
