# Round MERGE — combine the measured wins: fp8 padded TDM + r16 c32 TDM, disable the r64 regression

You are the ROUND-MERGE optimizer. **Time-box ~20 minutes** (integration, not research). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-merge`
(branch `qsa-index-scorer/opt-merge`, base = `bef193aa` = TDM-4: fp8 r32 144 B-pitch TDM + bf16 r64 shared-K
prefill). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`,
`records/` except `records/g2/`) or `records/sealed/`.

## Measured facts (root, a07-3 idle windows, correctness 22/22 on each tree)

- TDM-4 (`bef193aa`) with auto default: fp8 prefill bucket 942.7 → **910.5 µs (−3.4 %, keep)**; bf16 prefill
  **regressed** (bm16 770.1 → 826.2, +7.3 %; bmge64 762.9 → 816.4, +7.0 %) because the r64 kernel is
  235 VGPR / 105 SGPR — **the r64 fast path must be off by default**.
- R16 (`f64eedaf`, branch `qsa-index-scorer/opt-r16`, based on the same champion `3e3555b3`) with forced
  r16 TDM c32: decode_bf16_c65536 92.2 → **79.7 µs (−13.6 %)**, draft_bf16_c65536 86.6 → **82.7 µs
  (−4.6 %)**, decode_bf16_c32768 66.9 → 68.7 (+2.7 %, noise); correctness 22/22. It changes the r16 TDM
  geometry from four wave-private 64-column tiles (69,632 B, 193 VGPR) to four 32-column tiles
  (34,816 B, 97 VGPR) and adds `QSA_BF16_R16_TDM_COLS` (default 32, 64 = legacy).

## Task

1. **Merge `qsa-index-scorer/opt-r16` into this worktree** (`git merge qsa-index-scorer/opt-r16`), resolving
   conflicts in the kernel and the `_g2` gates so that both feature sets are present: fp8 r32 padded TDM
   (144 B pitch, descriptor encoding 4/3, 9,216 B LDS) from TDM-4, and the r16 c32 padded TDM geometry
   (32-column wave-private tiles, 34,816 B LDS) from R16.
2. Re-route the defaults to exactly the measured wins:
   - `auto` (default): bf16 **r32 prefill** TDM (unchanged), bf16 **r16 decode/draft → TDM c32**
     (previously raw; this is the measured −13.6 %/−4.6 % win), **fp8 r32 prefill → padded TDM**
     (unchanged from TDM-4), fp8 r16/decode → raw.
   - **r64 fast path off by default**: available only under an explicit opt-in switch
     (e.g. `QSA_BF16_R64=1`); default routing must keep r32 for prefill regardless of `rows >= 64*requests`.
   - `QSA_BF16_K_LOAD=raw` still forces the old all-raw behaviour; `tdm` forces the TDM variants as before.
   - Keep all safety guards, fallbacks (unaligned cache, partial/mixed blocks), wrapper API, i64
     descriptor extents, clamped/predicated stores and exactly-once logit writes.
3. Rebuild all CPU gates and the no-device ISA evidence for the merged tree: contract, trace16/32, address,
   Q-LDS, padded-layout/K-TDM byte tests (c32 and r32/fp8/r64 geometries), routing tests (updated to the new
   defaults, including r64 opt-in and raw/tdm overrides), strict TDM ISA, and the ISA routing equalities
   (auto-r16-c32 == forced-tdm c32; auto-r32 == forced tdm r32; fp8 auto == forced fp8 tdm; r64 only under
   opt-in). Report LDS bytes / VGPR / pitch / tensor/wait/ds_load_b128 counts and zero waterfall markers for
   every affected kernel.

Commit on `qsa-index-scorer/opt-merge`; report JSON {status, base_commit, commits[] (include the merge
commit + any routing commit), files_sha256, what_changed, routing_table{auto/raw/tdm numbers}, tdm_evidence,
expected_gain{case:% from the measured tables}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when
done; a clean merge + re-route + all gates green is the whole deliverable.
