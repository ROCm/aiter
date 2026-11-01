# Round CVTDEF — make the native fp8 conversion the default (QSA_FP8_CVT=fast)

You are the ROUND-CVTDEF optimizer. **Time-box ~12 minutes. Default flip only.** Base = `4a5dff18`
(round-FP8CVT branch `qsa-index-scorer/opt-fp8cvt`: native `v_cvt_scale_pk8_bf16_fp8` conversion path behind
`QSA_FP8_CVT=fast`, default `current`; kernel sha `ab5fd6d2…`). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-cvtdef`
(branch `qsa-index-scorer/opt-cvtdef`). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`,
`reference/`, `analysis/`, `records/` except `records/g2/`) or `records/sealed/`.

## Measured by root on a07-3 (idle windows, busy 0–3; GPU correctness 22/22 PASS with `QSA_FP8_CVT=fast`)

Paired fresh-process A/B `current` vs `fast` (3 legs each): **prefill512_fp8_c65536 158.9 → 147.6 µs (−7.1 %)**;
fp8 decode/bf16 paths unchanged (their deltas are idle-window noise). The fast path is bit-identical to the
current conversion in the emitted preserve-denormal mode (all 256 E4M3 codes, NaN/±480 endpoint handling,
signed zero, scales; 11 exhaustive CPU tests + 512 actual TDM fragments), uses 168 VGPR vs 193, and leaves
46 switch-off ISA files byte-identical.

## Task

Flip the default: `QSA_FP8_CVT` default **`fast`** (values `current|fast` still accepted, `current` forces the
old path for A/B). Everything else unchanged: only the fp8 r32 padded-TDM single-request fragment conversion
uses it; fp8 r16/raw/scalar/bf16 paths untouched; geometry/masks/descriptors/i64 extents/scales/head-sum/stores
identical. Update the CPU/ISA gates and routing/env-validation tests to assert the new default (default ==
explicit `fast`; `current` still yields the old ISA byte-identically; every unaffected kernel unchanged), and
regenerate the fp8 ISA evidence for the default. No-device stock-flydsl verification only (CPU-only container,
image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no `--device`; SSH user `blyu`, host
`heliosr-1b114-a07-3.mnb.dcgpu`, key `C:\Users\blyu\.ssh\id_ed25519`, staging dir
`/data/blyu/qsa/work/opt-cvtdef-cpu-trace-20261101`).

Commit on `qsa-index-scorer/opt-cvtdef`; report JSON {status, base_commit, commits[], files_sha256,
what_changed, routing_table{default/current}, isa_evidence{fast ISA sha, current ISA sha, unaffected identity},
expected_gain{fp8 prefill −7.1 % measured}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done.
