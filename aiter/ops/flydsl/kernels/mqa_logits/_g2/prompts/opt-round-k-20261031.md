# Round K — fp8 prefill: port the 32-row M tile (K reuse) to the fp8 WMMA path

You are the ROUND-K optimizer. **Hard time-box: ~25 minutes; commit a clean slice as soon as it is green;
if it cannot be made safe, commit nothing and leave the worktree clean.** No GPU; you may SSH to a07-3 for
CPU-only no-device tracing/ISA dumps. Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-k`
(branch `qsa-index-scorer/opt-k`, base = champion **`e0771e2a`**; kernel sha `90aaa1a0…`). Never push; never
touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/` except
`records/g2/`) or `records/sealed/`.

## Champion context (measured on a07-3; correctness 22/22 + 9/9 bucket shapes PASS)

`e0771e2a` = baseline + bf16 WMMA 16x16x32 + fp8 LDS-Q + fp8 WMMA + waterfall-loop fix + **bf16 32-row M
tile for prefill** (round J). Measured medians: decode_bf16_c65536 92.9 µs, decode_bf16_c32768 72.6,
draft_bf16 86.9, **prefill512_bf16 162.0**; decode_fp8_c65536 76.9, decode_fp8_c32768 69.5, draft_fp8 84.5,
**prefill512_fp8 192.0**.

Round J's bf16 change (see commit `e0771e2a`, branch `qsa-index-scorer/opt-j`, worktree `..\qsa-opt-j`):
a wave processes **two 16-row WMMA M tiles of the same request (32 rows)** with one K gather per step shared
by both tiles; selected only when `rows >= 32*num_requests and rows >= 32`, with the exact 16-row path as the
runtime fallback for mixed-request/tail/partial blocks (bitwise-equal). It is **bf16 only**; the fp8 WMMA
builder/dispatch was deliberately left unchanged.

## Task

**Port the same 32-row M-tile K-reuse design to the fp8 WMMA builder** (`_build_wmma_fp8_kernel`), keeping
the fp8-specific parts identical: two `buffer_load_b64` E4M3 K loads per WMMA step, the exact
E4M3→bf16 conversion, and the post-GEMM per-column fp32 scale in the scalar path's order. Add a
compile-time 32-row fp8 specialisation (kernel name suffix e.g. `_r32_c64`), select it with the same
`rows >= 32*num_requests and rows >= 32` rule in the wrapper's fp8 dispatch, and fall back to the exact r16
fp8 path for mixed-request/tail/partial blocks (bitwise-equal). Everything else (bf16 paths, scalar fp8
fallback, waterfall-free `_plane_bytes` descriptors, clamped predicated stores, i64 extents, host guards,
wrapper API, `QSA_FP8_PATH=scalar`) must stay unchanged.

Verification (CPU/offline): champion gates green — `test_contract_cpu.py` (incl. WMMA lane emulator and fp8
WMMA emulator), `trace_check_cpu.py` 16/16, `audit_addressing_cpu.py`, `test_q_lds_cpu.py`,
`test_fp8_wmma_cpu.py`, descriptor tests — plus new checks that the fp8 r32 fast path equals the r16
fallback bitwise on all fp8 fixtures, mixed-request 32-row groups fall back, rows∤32 tails work, and scale
application order is unchanged. `COMPILE_ONLY=1` trace of all specialisations must `[COMPILED]` (SSH recipe:
user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key `C:\Users\blyu\.ssh\id_ed25519`, unique staging dir
`/data/blyu/qsa/work/opt-k-cpu-trace-20261031`, image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no
`--device` flags, no gpu-lock).

Commit on `qsa-index-scorer/opt-k` and return JSON {status, base_commit, commits[], files_sha256,
what_changed, expected_gain{case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done.
