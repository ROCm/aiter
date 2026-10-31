# Round C — port the validated MFMA QSA kernel to our page-4 task (bf16 path) — brief

You are the ROUND-C optimizer (disposable, fresh context). You have **no GPU**; the root measures. Work only in
your own git worktree: `C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-c`
(branch `qsa-index-scorer/opt-c`, based on champion `1644eeb5`). Never push. Never touch frozen dirs
(`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/` except `records/g2/`) or
`records/sealed/`.

## Goal (highest-leverage available): replace our scalar bf16 path with the matrix-core design

Our current kernel `aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py` is scalar VALU,
one thread per output column: 572.5 µs for decode_bf16_c65536 (64 rows × 65536 cols), 196.8 µs draft,
3932.8 µs prefill512. The task package ships a **validated AMD FlyDSL MFMA implementation of the same math**
at `reference/aiter_kernels/flydsl__mqa_logits__qsa_flydsl_layout.py` (MIT header, 536 lines, gfx1250,
`fx.make_mma_atom(fx.rocdl.MFMA(16,16,16, fx.BFloat16))`, 256-thread blocks, BM=BN=64, Q kept in VGPRs,
K staged in LDS with row stride HDIMS+8 against bank conflicts, request-range wave reduction, early-exit).
A near-identical copy is at `golden_baseline/ATOM_MAIN_qsa_flydsl_layout.py`.

**Port that design into our kernel file for the bf16 path.** Keep our fp8 path exactly as it is (scalar; do
not regress it). Adapt the page geometry:

- Reference uses `KV_CACHE_BLOCK_SIZE=64`, `COMPRESS_RATIO=4` ⇒ `PAGE_SIZE = 16` compressed columns per
  physical page, cache `[pages, 16, 1, 128]`, block = 4 pages × 16 columns.
- **Our task uses page size 4** (`qsa_shapes.COMPRESSED_PAGE = KV_BLOCK // COMPRESS_RATIO = 16//4 = 4`):
  cache `[pages, 4, 1, 128]`, `logical_page = c // 4`, `page_off = c % 4`, `page_table[req, logical_page]`,
  `visible = clamp(min((pos+1)//4, ctx//4), 0)`. A 64-column block therefore spans **16 physical pages × 4
  rows** (not 4 pages × 16 rows). Adapt the K-gather/index code: page0 = col0/4, pages page0..page0+15, each
  page contributes 4 consecutive key rows (`cache[phys, off, 0, :]`, off = 0..3); mask `logical_page <
  page_table.shape[1]`, `phys in [0, num_pages)`, and `col < columns` for tail blocks.
- Q layout is unchanged: `q[rows, 4, 128]` bf16 contiguous; the reference's Q permutation (lane group g,
  k-step s → dims 32g+4s+[0..3]) and `make_fragment_A/B/C` usage can be reused as-is.
- Keep the reference's LDS struct/`fx.SharedAllocator().allocate(LDS).peek()` pattern, `fx.gpu.barrier()`
  after LDS stores, the early-exit `if col0 < vis_max`, the wave-0 metadata reduction, and the
  one-accumulator page-by-page MFMA loop. `rows` is runtime (4…2880); prefill buckets are e2e chunk rows
  (512-row chunks as measured by our harness).

## HARD safety constraints (learned the hard way on this task)

1. **No sentinel/OOB stores.** The reference's `gstore(...)` uses `(pred).select(offset, 0x7FFFFFFF)`
   sentinel stores. On gfx1250 the V# has **no bounds checking** (`OOB_SELECT=2` on RDNA) — such stores are
   real out-of-buffer accesses and previously produced `VM_PAGE_FAULT`/GPU hangs. Replace every sentinel
   store with an **in-bounds predicated store** (or a clamped index, like our `6c272c3c` contract): logits
   store index clamped to `[0, rows*columns)`, visible/row_starts stored only by the owning lane with an
   in-bounds index. No store address may leave its plane.
2. Keep every load in bounds too (page masks, `cols < columns`, row < rows, req < num_requests; metadata
   loads clamped to rows-1 as the reference does).
3. **Stock flydsl 0.3.3.dev886 only** (the image's): no `rocdl.sched.vgpr`/dev1004 APIs. The reference API
   surface is verified to exist: `fx.SharedAllocator().allocate(struct).peek()`, `fx.gpu.barrier()`,
   `fx.ptr_load(ptr, result_type=None)`, `fx.ptr_store(value, ptr)`, `fx.memref_load_vec/store_vec`,
   `fx.make_mma_atom(fx.rocdl.MFMA(16,16,16, fx.BFloat16))`, `fx.rocdl.make_buffer_tensor/make_buffer_ptr`.
   (Confirmed on the measurement node's image in a CPU-only container.)
4. Keep our public wrapper **API and semantics** exactly: `qsa_paged_mqa_logits_page4(q,
   compressed_k_cache, page_table, token_to_request, query_positions, context_lens, compress_ratio=..., 
   max_columns=..., compressed_k_scale=..., ...)` → `(logits [rows, columns] fp32, visible_groups [rows]
   int32)`, with `-inf` for every `c >= visible` and for invalid-page columns, and host-side guards from
   `6c272c3c` (alignment/dtype/shape/no-silent-int64-narrowing). bf16: `compressed_k_scale` is a dummy
   tensor (ignore). fp8: keep the existing scalar path unchanged.
5. bf16 logits outside `[0, visible)` must be `-inf` (our harness checks `outside_not_neg_inf == 0`): either
   have the kernel write `-inf` for masked columns or pre-fill the logits tensor with `-inf` before launch
   (document the extra memset cost if you choose that).

## Verification you must do offline (no GPU)

- `python3 _g2/test_contract_cpu.py` green (extend with cases for the new bf16 path: page-4 indexing,
  tail columns (1000), visible=0, t2r invalid, page OOB → -inf, rows not multiple of 64).
- `python3 _g2/audit_addressing_cpu.py` must still pass (update it to the new mapping if needed).
- Optional but valuable: `python3 _g2/trace_check_cpu.py`-style symbolic check of the new body.
- `COMPILE_ONLY=1` path (`_g2/compile_only.py`) must trace both dtype specialisations (tracing-only, not a
  real build — note that in the report).

## Deliverable

Commit(s) on `qsa-index-scorer/opt-c` (never push) with a compact JSON report:
`{status, base_commit, commits[], files_sha256, what_changed, expected_gain{case:%}, risks,
correctness_evidence{cpu_tests}, gpu_plan}`. If the full port is too large to land safely in one pass,
land the smallest valuable slice (e.g. bf16 decode/draft shapes with columns multiple of 64, keeping the
scalar path for the rest via a host-side dispatch) and say exactly what is covered. Report when done.
