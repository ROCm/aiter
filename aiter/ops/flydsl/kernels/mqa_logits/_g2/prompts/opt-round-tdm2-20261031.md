# Round TDM-2 — land the 80% gate: conflict-free LDS + TDM for the K gather (bf16)

You are the ROUND-TDM2 optimizer. **Time-box ~35 minutes.** Extend the existing TDM slice on
`qsa-index-scorer/opt-tdm` and deliver a clean CPU/ISA-gated commit; the root measures on a07-3.

Worktree: `C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-tdm2`
(branch `qsa-index-scorer/opt-tdm2`, base = `45c6c26c` = champion `eba80298` + first TDM slice). Never push;
never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/` except
`records/g2/`) or `records/sealed/`.

## Context: the gate is now measured and demanding

a07-3's real bandwidth ceiling (fw-bringup/ubench STREAM, idle window) is **read-only ≈ 19.6 TB/s**,
memcpy ≈ 15.3 TB/s → **80% roofline gate = 15.7 TB/s read-roofline** (see
`records/g2/roofline-gate-a073.md`). Current champion achieved bandwidth is only 1.0–5.9 TB/s (all 9 buckets
FAIL; best prefill_bf16_bm16 at 37.6% of gate). Per-lane 16-byte MUBUF loads of random 256-byte K rows give
~16–20% HBM efficiency; TDM can issue aligned 256-byte row fetches, which is the intended fix.

The first TDM slice (`45c6c26c`, opt-in `QSA_BF16_K_LOAD=tdm`, full single-request bf16 **r32** blocks only)
compiles to real ISA (2× `tensor_load_to_lds`, `s_wait_tensorcnt`, no waterfalls) but measured **+27 %
(prefill512 183→233 µs) / +40 % (2880-row 660→925 µs) slower** — because the shared 64×128 bf16 LDS tile is
consumed with a row stride of 256 B = 64 dwords: for a given k-chunk, the 16 lanes that read rows 0..15 hit
the **same bank group (16-way conflict)**.

## Your tasks, in priority order

**A. Fix the LDS bank conflicts (must).** Use the TDM descriptor's **padding** fields to give the LDS rows a
padded stride, then adapt the consumer to the padded layout. Suggested target: row stride **272 B**
(68 dwords; pad after 64 dwords with 4 dwords) → row starts mod 128 B become 0,16,32,…,240 → banks spread,
~2-way instead of 16-way conflicts. Verify the exact `pad_interval` / `pad_amount` encoding in
`flydsl/expr/rocdl/tdm_ops.py` (there is a `compute_padding_encoding` helper; the kernel-wiki page
`MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-tdm-programming.md` says: pad_interval encodes the number
of DWORDs stored before padding (0=2 … 7=256, i.e. log2-like), pad_amount = padding DWORDs to add (0=1 …),
both zero or both non-zero, tile_dim0 multiple of 4 B, padding skips LDS locations). Keep every 16-byte
consumer access aligned. Update `_pair_base_tdm` / `_pair_frag_tdm` (lo at +0, hi at +32 within the row) and
the LDS allocation size (`64 * 272 B` rounded up), or implement the equivalent explicit padded layout. Keep
the descriptor's `row_ids` semantics and the raw fallback unchanged.

**B. Extend TDM staging beyond the single r32 case (should).** Today only full single-request bf16 r32 blocks
use TDM; decode/draft and r16 blocks still use per-lane loads. Extend the TDM gather to:
  - bf16 **r16** blocks (decode/draft): each wave's 64-column K tile = 64 rows × 256 B = 16 KiB; per request
    pass, gather the 64 rows with 8 TDM ops (8 i32 ids each) into the (padded) LDS tile, `tensor_wait(0)`,
    then consume as above; the block's ≤4 request passes need the tile reloaded per pass (serialize with the
    barrier/wait discipline; note per-wave issue-to-XACK limits 3/wave, completion limit 63).
  - Keep raw fallback for unaligned cache bases / mixed or partial blocks if the guard semantics get
    complicated; correctness first. fp8 may stay raw this round.
**C. If time permits:** double-buffer the LDS tile across request passes (issue next pass's TDM while the
current pass computes) and/or apply the same padded layout to the fp8 WMMA path's K loads.

## Hard constraints

Stock flydsl **0.3.3.dev886** only. LDS: one `SharedAllocator` per kernel, 16-byte alignment minimum; TDM
completion is `tensor_wait(0)` + workgroup barrier before cross-wave LDS consumption (XACK is NOT
LDS-visibility); same-wave TDM ops ordered; tensor ops are uniform/SGPR-state; gather row ids must be valid
(clamp holes/tails to physical page 0 as the current slice does). Keep the existing safety contract,
`_plane_bytes` i64 extents, clamped/predicated stores, wrapper API and host guards. Default path must stay the
current champion behaviour unless you can prove the TDM path correct on all fixtures (the wrapper must route
unsupported shapes to raw).

## Verification (no GPU) and deliverable

- Read `tdm_ops.py` on the node (CPU-only container, image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no
  `--device`; SSH user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key `C:\Users\blyu\.ssh\id_ed25519`,
  unique staging dir `/data/blyu/qsa/work/opt-tdm2-cpu-trace-20261031`).
- Real ISA evidence (`COMPILE_ONLY=1` + `FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=...`):
  `tensor_load_to_lds`, `s_wait_tensorcnt`, padded LDS size, `ds_load_b128` consumers, no
  `v_readfirstlane_b32`/`v_max_i64`; count and report the padded stride.
- CPU gates green: existing contract/trace16/address/Q-LDS, plus a layout test proving the padded LDS tile is
  byte-exact vs the raw path and that the consumer reads the right element for every lane/k-chunk (including
  subnormals/NaNs unchanged), and that padded-stride LDS accesses stay in bounds.
- Commit on `qsa-index-scorer/opt-tdm2`; report JSON {status, base_commit, commits[], files_sha256,
  what_changed, tdm_evidence{padding fields/stride/ISA markers/lds bytes/waits}, expected_gain{case:%},
  risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done (prefer a clean partial result).
