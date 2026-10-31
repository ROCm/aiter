# Round I — kill the MUBUF waterfall loops (64-bit `v_max_i64` descriptor prologue)

You are the ROUND-I optimizer (disposable, fresh context). **No GPU**; the root measures. Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-i`
(branch `qsa-index-scorer/opt-i`, base = **commit `02b0d923`** = champion G: bf16 WMMA + fp8 WMMA).
Never push. Never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/`
except `records/g2/`) or `records/sealed/`.

## The finding you must fix (proved by a real backend ISA dump, no GPU needed)

Round G's no-device trace with `COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=...`
emits the **real backend ISA** (`dump/<kernel>/21_final_isa.s`, `20_llvm_ir.ll`). In champion G/F's
`bf16_wmma` ISA, **every buffer access is wrapped in a waterfall loop**:
`912 v_readfirstlane_b32` / `456 v_cmpx_eq_u64` / `228 s_and_not1_wrexec` self-loops for 164 buffer ops, each
with `s_wait_loadcnt 0x0` inside the loop ⇒ all loads serialized, no memory-level parallelism.
Same pattern in the scalar bf16 (87 loops) and fp8 (25) kernels.

Root cause: the descriptor/plane helper (`_plane()` / `_ld_vec` call sites) computes
`_imax(fx.Int64(num_bytes), fx.Int64(4))`. LLVM lowers `llvm.smax.i64` to `v_max_i64` (no SALU 64-bit max) ⇒
the resulting V# is materialized in VGPRs ⇒ MUBUF rsrc legalization inserts a per-access waterfall loop.
**Likely fix** (keep both the i64 extents and the `>= 4`-byte guarantee): clamp the **i32** factors with
`s_max_i32` *before* widening, e.g. `fx.Int64(_imax(num_bytes_i32, 1))` / `Int64(rows) * Int64(1024)` etc.,
so no 64-bit max is emitted and the V# stays in scalar registers. Round G is applying this to the new
fp8_wmma kernel behind a build-time switch; your job is to apply the same fix to the **bf16_wmma path first**
(the biggest measured case: decode_bf16_c65536 97.1 µs / prefill512_bf16 216.6 µs) and then, if trivially safe,
to the scalar bf16 and fp8 paths.

## Required verification (all offline, no GPU)

1. Reproduce the ISA dump yourself on a07-3 (CPU-only container, stock flydsl **0.3.3.dev886**, image
   `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no `--device` flags, no gpu-lock; SSH: user `blyu`, host
   `heliosr-1b114-a07-3.mnb.dcgpu`, key `C:\Users\blyu\.ssh\id_ed25519`, unique staging dir
   `/data/blyu/qsa/work/opt-i-cpu-trace-20261031`):
   `COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=<dir>
   python3 _g2/compile_only.py`
   and confirm the dump path/format. **Before/after evidence is required**: count
   `v_readfirstlane_b32` / `s_wait_loadcnt` / waterfall self-loop labels in `21_final_isa.s` for the
   `bf16_wmma` kernel; after your fix the count of `v_readfirstlane_b32` and the waterfall self-loops around
   `buffer_load` must drop to ~0 (some may remain for genuinely divergent cases — count and explain).
2. Keep every existing CPU gate green: champion-G `test_contract_cpu.py` (incl. WMMA lane emulator),
   `trace_check_cpu.py` 16/16, `audit_addressing_cpu.py`, `test_q_lds_cpu.py`, `test_fp8_wmma_cpu.py`.
3. `COMPILE_ONLY=1` trace must still `[COMPILED]` all specialisations (bf16, fp8, bf16_wmma, fp8_wmma).
4. Source-level safety must be unchanged: the >=4-byte descriptor guarantee, i64 extents passing to the
   backend, in-bounds clamped predicated stores, wrapper API/host guards.

Commit on `qsa-index-scorer/opt-i` with a compact JSON report: {status, base_commit, commits[],
files_sha256, what_changed, isa_evidence{before/after counts, dump paths}, expected_gain{case:%}, risks,
correctness_evidence{cpu_tests}, gpu_plan}. Never push. Report when done.
