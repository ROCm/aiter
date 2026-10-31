# Round TDM — use the gfx1250 Tensor Data Mover for the K gather (80%-roofline push)

You are the ROUND-TDM optimizer. **Hard time-box: ~20 minutes.** Deliver a clean, gate-verified slice or
commit nothing and report feasibility findings. No GPU (root measures); you may SSH to a07-3 for CPU-only
no-device tracing/ISA (FlyDSL `COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=...`).

Worktree: `C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-tdm`
(branch `qsa-index-scorer/opt-tdm`, base = champion `eba80298`, kernel sha `057fbc4f…`). Never push; never
touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/` except `records/g2/`)
or `records/sealed/`.

## Why: the human says MI450 kernels should use TDM, and a07-3 must meet an 80 % roofline gate

Current champion (measured a07-3, idle windows): decode_bf16_c65536 89–93 µs, prefill512_bf16 161–162,
prefill512_fp8 163–164, decode_fp8 77; bucket-modeled bandwidth is only ~16 % / ~30 % / ~10 % of the
20 TB/s fallback peak — far under the 80 % gate. The K data is 256 B rows gathered by page-table physical
index; today each access is a per-lane `buffer_load` (the waterfall loops were removed, but still per-lane
MUBUF). gfx1250 has a **TDM (Tensor Data Mover)** descriptor-driven DMA that moves structured tiles
Global→LDS asynchronously, including a **gather mode** for random row indices — exactly this pattern.

**FlyDSL stock 0.3.3.dev886 DOES expose TDM** — read these on the node (image
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, CPU-only container, no --device):
`/usr/local/lib/python3.12/dist-packages/flydsl/expr/rocdl/tdm_ops.py` (API: `make_tensor_descriptor_2d`
with `global_ptr, lds_memref, global_offset, tensor_shape, strides, tile_shape, elem_bytes, pad_interval,
pad_amount, num_warps`; `tensor_load_2d(desc)`; `tensor_load_gather`; `tensor_store_2d`;
`tensor_wait(N)`; also raw ops `tensor_load_to_lds` / `tensor_store_from_lds` with D# groups in
`flydsl/expr/rocdl/__init__.py`). The kernel-wiki pages that document it (internal, do not copy text):
`MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-tdm-programming.md`, `mi400-tdm-sync.md`,
`mi450-bottleneck-tdm-async.md` — key rules: one instruction per tile, EXEC ignored; D# is SGPR state
(uniform); issue-to-XACK limit 3/wave, 6/SIMD; completion limit 63; `S_WAIT_TENSORCNT` waits for
**completion** (XACK ≠ data in LDS); same-wave TDM ops are ordered; tensor ops never in a clause;
direct-copy needs 128 B/256 B fetch granularity and ≥128 B alignment; tile_dim0 must be 4-byte multiple.

## Task (smallest valuable TDM slice)

In the bf16 WMMA kernel (64-column N tile), load the K rows for one column tile via **TDM gather into LDS**
instead of per-lane MUBUF loads, then consume the B fragments from that LDS tile:
- K row for logical column c = `compressed_k_cache[page_table[req, c//4], c%4, 0, :]` = 256 contiguous bf16
  bytes; build a 2D tensor descriptor over the flattened cache rows `(num_pages*4, 128)` (strides `(128,1)`,
  `elem_bytes=2`, tile `(64, 128)` = 16 KiB LDS) and use the gather/row-index form (`tensor_load_gather`) with
  the page-table-derived row ids (phys*4 + c%4) — or, if the gather form is awkward, a 2D load per page
  (4 contiguous rows = 1 KiB) is an acceptable first slice.
- Wait with `tensor_wait` before consuming; LDS tile → WMMA B fragments (you may reuse the CDMA reference's
  `thr_mma.make_fragment_B(lds_view)` pattern where the LDS layout matches; otherwise copy LDS→registers with
  plain LDS reads and keep the existing fragment construction).
- Keep the current non-TDM path as the runtime fallback (env switch or shape guard) so correctness holds if
  TDM lowering has issues; all existing CPU gates and the 22-case GPU-correctness contract must stay green.
- Safety unchanged: in-bounds clamped/predicated stores, i64 descriptor extents, `_plane_bytes` form, host
  guards, wrapper API. fp8 path may stay as-is for this round.

## Verification (no GPU)

- Read `tdm_ops.py` and prove a minimal TDM kernel traces: `COMPILE_ONLY=1` + ISA dump must show
  `tensor_load_to_lds` / `TENSOR_LOAD_TO_LDS` and no v_readfirstlane/v_max_i64 waterfall signatures; all
  specialisations `[COMPILED]`.
- Keep existing CPU gates green; add a test that the TDM-loaded K tile equals the per-lane-loaded K tile
  bitwise (Python-level model) and that the LDS layout/consumer matches.
- Commit on `qsa-index-scorer/opt-tdm` **only if green**; report JSON {status, base_commit, commits[],
  files_sha256, what_changed, tdm_evidence{isa markers, descriptor fields, lds bytes, waits}, expected_gain{
  case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}. If the slice cannot be landed safely in the
  time-box: commit nothing, leave the worktree clean, and report exactly what blocked it (API signature,
  lowering gap, layout mismatch) — that feasibility answer is itself valuable.
