# Round FP8CVT — cut the fp8 prefill conversion cost (VALU/LDS bound) on champion 1ed0581a

You are the ROUND-FP8CVT optimizer. **Time-box ~45 minutes.** Base = champion `1ed0581a` (kernel sha
`bcce45a2…`). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-fp8cvt`
(branch `qsa-index-scorer/opt-fp8cvt`). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`,
`reference/`, `analysis/`, `records/` except `records/g2/`) or `records/sealed/`.

## Why (measured)

fp8 prefill (r32 single-tile padded TDM) is **903.8 µs** for the bmge64 bucket = 8.1 % of the 15.7 TB/s gate.
Its actual traffic (~0.8 GB K + 0.76 GB logits) is only ~1.7 TB/s effective, while the fp8 WMMA compute floor
is ~51 µs — so the kernel is dominated by **per-element E4M3→f16 conversion and LDS/VALU work**, not HBM.
Note the persistent fp8 variant (round FP8P) measured *worse* (P4 +24 %, P8 +46.5 %): fewer, fatter blocks lose
parallelism, so **keep the single-tile geometry** and attack the conversion path instead.

## Task

Reduce the fp8 K conversion cost in the existing fp8 r32 padded-TDM prefill kernel, opt-in (e.g.
`QSA_FP8_CVT=fast`, default current behaviour):
- inspect the real ISA of the current fp8 r32 kernel and count the conversion instructions
  (`v_cvt_*f16_fp8*` / `v_cvt_f32_fp8` / packing ops) per K element; report the current instruction count per
  chunk per wave;
- try the cheapest exact alternative(s), in order of preference:
  1. use **hardware packed fp8→f16 conversion instructions** (e.g. `v_cvt_pk_f16_fp8`-style, two or more
     elements per instruction) if the gfx1250 ISA/LLVM exposes them through inline assembly/flydsl intrinsics;
  2. widen the conversion path to process 4/8 fp8 elements per instruction via vectorised codegen
     (`uint32` → 4×f16 with `v_cvt_*` pairs) while preserving bit-exact E4M3 semantics (denormal mode that
     the current kernel uses — fp16 denormal-preserve — must be reproduced exactly);
  3. if neither is exact, keep the current conversion and instead hoist/reuse the converted B fragments
     across the two M tiles if that saves work (state the measured ISA count before/after).
- the fast path must be **bit-identical** to the current fp8 output (all 256 E4M3 codes, subnormal/NaN
  behaviour, scales, head-sum order); a CPU model proving this is mandatory. Keep the switch-off tree
  byte-identical to the champion and do not touch bf16/r16 paths.

## Hard constraints

Stock flydsl 0.3.3.dev886 only; no changes to the WMMA fragment layout, TDM descriptor or masks beyond what
the conversion path needs; keep the safety contract (`_plane_bytes` i64 extents, clamped/predicated stores,
host guards, wrapper API, every logit written once).

## Verification (no GPU) + deliverable

- All CPU gates green (contract/address/Q-LDS/trace16+32/K-TDM/fp8 K-TDM/routing/BM64/persist/strict ISA) plus
  new exhaustive fp8-code conversion tests (all 256 codes × both halve positions × subnormal modes) proving
  bit-identity vs the current path; switch-off ISA byte identity.
- No-device real-ISA trace (CPU-only container, image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no
  `--device`; SSH user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key
  `C:\Users\blyu\.ssh\id_ed25519`, staging dir `/data/blyu/qsa/work/opt-fp8cvt-cpu-trace-20261101`): report
  conversion instruction counts before/after per chunk/wave, VGPR/SGPR, WMMA/LDS counts, zero waterfall
  markers, dispatch names.
- Commit on `qsa-index-scorer/opt-fp8cvt`; report JSON {status, base_commit, commits[], files_sha256,
  what_changed, isa_evidence{conversion insns before/after, resources}, expected_gain{case:%}, risks,
  correctness_evidence{cpu_tests}, gpu_plan}. If no exact faster conversion exists, deliver a clean negative
  report with the ISA evidence (that is a valid outcome).
