# CK FMHA Split-KV Benchmark - Work in Progress

## Goal

Replace Triton unified attention for decode with CK FMHA split-KV (via
`mha_varlen_fwd` with `block_table`), which supports sliding window, sinks,
and block_size=32 — features the current Triton path handles but CK's
`42_unified_attention` kernel does not.

## Branch Layout

| Branch | Base | Purpose |
|--------|------|---------|
| `aghamari/bench-ck-fmha-splitkv` | `main` | **This branch.** Benchmark + fixes on main's CK (`b0c13f312`). |
| `aghamari/ck-unified-attention-rebased` | custom CK fork | Original unified-attention work (stashed changes). |

## What Was Done

### 1. Benchmark Script (new file)
`op_tests/triton_tests/attention/bench_ck_vs_triton_from_jsonl.py`

Benchmarks CK FMHA split-KV vs Triton 2D/3D on decode shapes from a JSONL
trace. Uses `mha_varlen_fwd` with `block_table` for CK, CUDA graph capture
for timing, per-shape CSV output.

### 2. Bug Fixes in `aiter/ops/mha.py`

- **`filter_fwd` typo (pre-existing on main):** In `cmdGenFunc_mha_varlen_fwd`,
  the `block_table is not None` branch used `filter_fwd` (undefined in that
  scope) instead of `filter_fwd_splitkv2` for logits filtering. Fixed.
- **Codegen filter too broad:** The combine-kernel filter `"*"` caused the
  codegen to generate d32 instances that have no matching combine kernel,
  triggering an assertion in `fmha_fwd_splitkv.py:1088`. Fixed by:
  - Adding `_d{hdim}` prefix to both combine and splitkv filters.
  - Adding `_nsink/_sink` suffix to the splitkv filter.
  - Using `--mask generic` instead of the default `simplified` (which also
    lacked combine-kernel matches).
- **Single blob gen command:** Replaced the two-command approach (separate `fwd`
  + `fwd_splitkv`) with a single `fwd_splitkv` command that generates both
  splitkv and combine kernels via the `@`-separated filter.

### 3. C++ Fix in `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu`

- **Relaxed `page_block_size` check:** Changed from `% 128 == 0` to
  `>= 16 && power-of-2`. The old check was overly restrictive; the kernel
  itself supports any power-of-2 block size >= 16.

## Current Blocker

After all the above fixes, the JIT module compiles the codegen blobs
successfully but **HIP compilation of the generated instances fails** with:

```
block_masking.hpp:113: GetTileRangeAlongX expects 3 args, got 5
```

This is a **bug in CK's `block_masking.hpp`** at commit `b0c13f312` (main's
submodule). The `GenericAttentionMask<true, true>` template path's
`GetTileRangeAlongX` has a 3-parameter signature, but the split-KV pipeline
at `fmha_fwd_splitkv_kernel.hpp` calls it with 5 arguments when paged-KV is
enabled.

This affects all `*_mg_*_pagedkv_*` (generic-mask + paged-KV) split-KV instances.

## How to Unblock

**Option A:** Update the CK submodule to a newer commit on `develop` that has
the `GetTileRangeAlongX` fix. Check:
```bash
cd 3rdparty/composable_kernel
git log develop -- include/ck_tile/ops/fmha/block/block_masking.hpp
```
Look for commits that change `GetTileRangeAlongX` to accept 5 parameters.

**Option B:** Use the `simplified` mask implementation instead of `generic`.
This requires fixing the codegen combine-kernel assertion for simplified mask +
pagedkv combos (the assertion at `fmha_fwd_splitkv.py:1088`). The fix would be
in the CK codegen to generate matching combine kernels for the simplified mask.

**Option C:** Fix `block_masking.hpp` directly — add the missing 2-parameter
overload or update the call site in the split-KV pipeline.

## Test Command

```bash
cd /workspaces/workspace/aiter
python op_tests/triton_tests/attention/bench_ck_vs_triton_from_jsonl.py \
    --jsonl /workspaces/workspace/aiter_unified_attention.jsonl \
    --warmup 5 --iters 20 --max-shapes 5
```

## Files Changed

- `aiter/ops/mha.py` — codegen filter fixes
- `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu` — relaxed page_block_size check
- `op_tests/triton_tests/attention/bench_ck_vs_triton_from_jsonl.py` — new benchmark
- `CK_FMHA_SPLITKV_BENCHMARK_STATUS.md` — this file
