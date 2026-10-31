# Round TDM-4 — fp8 padded TDM + bf16 BM=64 shared-K prefill (on champion 3e3555b3)

You are the ROUND-TDM4 optimizer. **Time-box ~40 minutes.** Base = champion
`3e3555b3` (bf16 r32 padded-TDM default, r16 raw; kernel sha `1ef645c1…`). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-tdm4`
(branch `qsa-index-scorer/opt-tdm4`). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`,
`reference/`, `analysis/`, `records/` except `records/g2/`) or `records/sealed/`.

## Established facts (do not re-derive)

- a07-3 measured limits: sequential STREAM 19.6 TB/s read-only; **random 256 B page-table K gather ceiling
  ~5.8 TB/s** (coalesced rows, max MLP) — the 80 % gate (15.7 TB/s) is unreachable for random-gather-bound
  buckets; the prefill buckets are the achievable target.
- TDM-2 found the stock `make_tensor_gather_descriptor` padding units are **bf16 element counts**
  (`compute_padding_encoding`: interval_dw = interval_elems*16//32); 128/8 → DWORD 64/4 → encoded 5/3 →
  **272 B LDS row pitch** for 256 B bf16 rows, consumer base `row*272`, lo +0 / hi +32 within the row.
- Current champion: bf16 r32 prefill stages a 64-row K tile with 2 TDM gather ops per wave, 17,408 B LDS,
  one barrier pair; measured −18.2 % prefill512, −19.5 %/−19.8 % on the bm16/bmge64 gate buckets.

## Tasks (priority order)

**A. fp8 prefill TDM** (decode_fp8/prefill_fp8 are still raw, 5–13 % of gate). E4M3 K row = 128 B; use a
**144 B padded LDS row pitch** (128 B data + 16 B pad; consumer base `row*144`, 16-byte aligned reads). Verify
the padding encoding for 1-byte elements from `flydsl/expr/rocdl/tdm_ops.py` source (the field units depend on
`elem_bytes`; compute the encoded interval/amount, don't guess). Apply it to the fp8 **r32** prefill path first
(same single-request guard), keep fp8 r16/decode raw, keep the raw fallback and the auto routing semantics
(bf16 behaviour unchanged).

**B. bf16 BM=64 shared-K prefill specialisation.** One workgroup covers **64 query rows × 64 columns of a
single request**; one padded 272 B K tile (17,408 B) is staged **once and shared by all four waves**, each
wave consuming 16 rows (one WMMA M tile) → K staged once per 64 rows, halving the K stream vs r32. Dispatch
it for prefill shapes with `rows >= 64*num_requests` (so the bmge64 bucket gets it); keep r32 for bm16-style
shapes and the raw fallback. If A and B both fit in the time box, prefer the one you can fully verify; state
what you did.

**C. Optional:** double-buffer the tile across the two M-tile groups if cheap.

## Hard constraints

Stock flydsl 0.3.3.dev886 only. TDM rules: 1 instruction per tile, uniform SGPR D#/row ids, tensor ops not in
clauses, `tensor_wait(0)` + workgroup barrier before cross-wave LDS consumption and before overwriting a
tile, per-wave issue-to-XACK ≤3, per-SIMD ≤6, completion ≤63; invalid row ids clamp to physical page 0. Keep
the safety contract (`_plane_bytes` i64 extents, clamped/predicated stores, host guards, wrapper API, every
logit written once), the bf16 auto routing, and default behaviour for paths you do not change.

## Verification (no GPU) + deliverable

- CPU gates green: existing contract/trace16+32/address/Q-LDS/padded-layout/strict-TDM-ISA/routing gates;
  add byte-exactness tests for the new fp8 padded tile and the BM64 tile geometry (every lane/k-chunk,
  invalid pages, tails, masks/stores-once, LDS bounds/alignment).
- No-device real-ISA trace on the stock image (CPU-only container, no `--device`; SSH user `blyu`, host
  `heliosr-1b114-a07-3.mnb.dcgpu`, key `C:\Users\blyu\.ssh\id_ed25519`, staging dir
  `/data/blyu/qsa/work/opt-tdm4-cpu-trace-20261031`): report per-kernel tensor/wait/ds_load128/LDS bytes/
  barrier pairs/pitch and zero waterfall markers.
- Commit on `qsa-index-scorer/opt-tdm4`; report JSON {status, base_commit, commits[], files_sha256,
  what_changed, tdm_evidence{geometry, pitch, LDS bytes, ISA counts}, expected_gain{case:%}, risks,
  correctness_evidence{cpu_tests}, gpu_plan}. Prefer a clean partial result; report when done.
