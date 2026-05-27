# CK Unified Attention — Work-Log + Test Status

_Last updated: 2026-05-27. Status owner: jukorhon._

This document tracks the current state of the CK unified-attention work
that lives on the `jukorhon/unified-attention-ck` branch (same name on
both repos for coherence):

| Repo                          | Branch                              | HEAD (short) |
|-------------------------------|-------------------------------------|--------------|
| `ROCm/aiter`                  | `jukorhon/unified-attention-ck`     | `e78240f74`  |
| `ROCm/composable_kernel`      | `jukorhon/unified-attention-ck`     | `2645149bb`  |

The aiter branch pins the CK submodule to the matching commit, so a
recursive checkout is enough.

---

## How to reproduce / test

### One-time checkout

```bash
git clone --recursive -b jukorhon/unified-attention-ck \
    https://github.com/ROCm/aiter.git
cd aiter

# Sanity check: the CK submodule should be on the matching branch.
cd 3rdparty/composable_kernel
git log -1 --oneline   # should print: 2645149bb CK-UA: shrink Tier-2 page-table LDS cache to per-split window
cd -

pip install -e . --no-build-isolation
```

### Canonical test script: `op_tests/test_unified_attention_ck.py`

This is the basic testing script — comprehensive grid (correctness vs
`ref_paged_attn` + perf vs Triton) **and** single-shape mode for ad-hoc
investigations.

```bash
# Default grid (~48 configs, a few minutes)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py

# Smoke subset (~8 configs, <1 min)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py --quick

# Full Triton-UA-style matrix (~288 configs, ~30 min)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py --full

# Restrict any dimension(s) of the grid
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py \
    --head-size 128 --dtype bf16 --q-dtype none

# Single-shape mode (replaces the standalone test_single_shape.py for
# routine correctness/perf checks). `--num-blocks auto` allocates a
# unique-per-token KV pool so block_tables hold no fake L2 reuse.
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py \
    -b 64 -sq 1 -sk 128000 \
    --num-heads 64,8 --head-size 128 --block-size 16 \
    --dtype bf16 --q-dtype none --num-blocks auto
```

Exit code is non-zero if any row's `ck_pass` or `triton_pass` is `False`,
so it's CI-ready.

### Specialised scripts under `ua-test-scripts/`

These pre-date the single-shape mode on the canonical script and stay
around for the specific things they do better than the comprehensive
runner. Default to `test_unified_attention_ck.py` unless you specifically
need one of these:

| Script                     | When to use it                                                                                                  |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| `test_single_shape.py`     | Kernel-tuning iteration loop: CUDA-graph timing with `BENCH_START_CK`/`BENCH_END_CK` rocprof markers; reports GB/s + TFLOPs in addition to time. Faster turnaround than the @perftest path when you're scrubbing a single shape repeatedly. |
| `regression_decode.sh`     | Multi-run median-of-N sweep across the 4 (d, dtype) decode combos; surfaces noise from shared-GPU runs.         |
| `probe_decode_d128.sh`     | One-off perf probes used during the Tier-2 / scalar-prefetch investigation; kept for historical reproducibility. |
| `probe_prefill_d128.sh`    | Same, prefill side.                                                                                              |
| `tier2_decode_sweep.sh`    | Tier-2 LDS cache investigation sweeps.                                                                            |
| `rocprof_analysis/`        | rocprof CSVs + analysis notebooks from the kernel investigation runs.                                            |

---

## Recent changes on the branch

### CK side — `jukorhon/unified-attention-ck`

```
2645149bb  CK-UA: shrink Tier-2 page-table LDS cache to per-split window   [LATEST]
badc80702  CK-UA: enable Tier-2 LDS page-table cache on decode + fix split-KV bulk-load OOB
310efc556  CK-UA: halve kBlockN for bf16/fp16 m16 decode + generalise PVAttrNumAccess
89b54563b  CK-UA: skip post-load page_offsets refresh on final K/V tile
06e1a70e7  CK-UA: constexpr page_size (Tier 3) — prefill_d64 fp8 -15.8%, prefill_d128 fp8 -6.3%
045b1f57b  CK-UA: widen FP8 K/V async loads to dwordx4 where the tile allows it
7a319d9a4  CK-UA: drop redundant phase-0 s_barrier (-3% fp8 prefill_d128 decode)
3431615ff  CK-UA: fuse FP8 cvt + cross-lane swap to hide ds_bpermute latency
9d7cc3ee9  CK-UA: extend FP8 to the 16x16x32 _m16 decode tier via LDS roundtrip
63c75277a  CK-UA: enable FP8 (e4m3) for prefill/m128 and the 32x32x16 small-tile decode variants
```

(For the full series back to the kernel-variant collapse, see
`git log --oneline jukorhon/unified-attention-ck`.)

### aiter side — `jukorhon/unified-attention-ck`

```
e78240f74  unified_attention_ck: bump CK + add decode regression script   [LATEST]
c3b09c3d7  unified_attention_ck: bump CK + add ua-test-scripts/ for shape-level testing
b63386f0d  unified_attention: bump split-KV target to 4x CUs, cap 128
278e72ffa  unified_attention_ck: bump CK submodule for block_tables OOB fix
30458aa15  Add correctness + perf tests for CK unified attention
b518460f9  Wire CK unified attention kernel into aiter
```

---

## Latest fix in detail: per-split Tier-2 page-table LDS window (CK `2645149bb`)

**Symptom.** Device-side assert
`assert(split_end_page <= kPageTableLdsEntries)` firing on long-context
decode at `page_size=16`, e.g.

```
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py \
    -b 64 -sq 1 -sk 128000 \
    --num-heads 64,8 --head-size 128 --block-size 16 \
    --dtype bf16 --q-dtype none --num-blocks auto
```

reported a "compilation error" by hipcc but was actually a runtime
assertion (the templated kernel signature in the message is enormous,
which masks that it's a `printf`+`abort` rather than a `clang`+`error:`).

**Root cause.** The Tier-2 LDS page-table cache bulk-loaded
`block_tables_ptr_[block_table_offset + i]` into `block_tables_lds[i]`
for `i ∈ [0, num_blocks)` — the **absolute** prefix of the batch's page
table, including pages earlier splits had already consumed. That made
the 4096-entry static cap a *total-sequence* cap rather than a
*per-split* one:

```
b=64 sk=128000 page=16
  → total_kv_pages = 8000
  → num_splits     = 4   (wrapper heuristic, ~4× CU count)
  → last-split window     = [6000, 8000)   (2000 pages, ≪ 4096)
  → loaded into LDS       = [0, 8000)      (8000 entries → assert fires)
```

The wrapper's split-KV scheduler was already keeping each CTA's working
set well under the cap — the kernel just needed to load the right slice.

**Fix.** Make the bulk load *per-split-window*: only load
`[split_start_page, split_end_page)` where
`split_start_page = ⌊num_blocks_start · kPageBlockSize / page_size⌋`,
and shift `refresh_{k,v}_offsets`' lookup by `split_start_page` so the
LDS index stays in `[0, split_window_pages)`. `split_start_page` is a
kernel-entry constant, so the subtract folds into the `s_load_dword`'s
immediate offset — no per-iteration cost.

On the prefill path (`num_blocks_start == 0`) `split_start_page == 0`
and the change collapses to the original absolute-indexed load, so
prefill codegen is bit-identical.

**Verified on gfx950 (MI355X, 256 CUs).**

| Shape                                                | Before          | After            |
|-------------------------------------------------------|-----------------|------------------|
| b=64 sq=1 sk=128000 d=128 bf16 page=16                | `assert` fired  | 5713 GB/s, PASS  |
| b=16 sq=1 sk=65536  d=128 bf16 page=16                | `assert` fired  | 5679 GB/s, PASS  |
| b=4  sq=1 sk=262144 d=128 bf16 page=16                | `assert` fired  | 5687 GB/s, PASS  |
| b=1  sq=1 sk=524288 d=128 bf16 page=16                | `assert` fired  | 5349 GB/s, PASS  |
| b=128 sq=1 sk=16384 d∈{64,128} dtype∈{bf16,fp8} ps=32 | baseline        | unchanged perf, PASS |
| b=128 sq=1 sk=16384 d∈{64,128} dtype∈{bf16,fp8} ps=16 | baseline        | unchanged perf, PASS |
| b=1  sq=4096 sk=4096 d=128 bf16 ps=32 (prefill)       | baseline        | bit-identical (codegen no-op), PASS |

Side effect: the bulk-load bytes for splits ≥ 1 also drop by the number
of pages we used to read but never index — small but free.

---

## Open / parked items

* **Hoisted SGPR-ring prefetch (Tier-0 → moved before `fmha_alu1`).** Tried
  and reverted. The hoisted variant replaced fast Tier-2 LDS broadcasts
  with multiple scalar global loads hitting L1, costing 300–400 cycles
  per iteration vs. ~40 baseline. Net regression of 4–12% across the
  decode regression set. Conceptually cleaner and would free ~16 KiB of
  LDS, but the cost model doesn't pay off until we have a way to keep
  the scalar load latency hidden. Parked.

* **Shrink Tier-2 cache per kernel-policy.** Tried sizing `kPageTableLdsEntries`
  down to 1024/2048 entries per `DecodePolicy` / `TinyDecodePolicy` so
  we get back ~12 KiB of LDS on the decode side. Triggered the AMD
  occupancy heuristic to drop VGPR count, net-regressing 10–25%
  (occupancy was already saturated; the heuristic doesn't know that).
  Could be retried with explicit `amdgpu_waves_per_eu` to pin occupancy
  while shrinking. Parked.

* **The 64-bit-base global_load_lds path** (`cache_ptr_int32_overflow_possible`):
  currently dispatch-only; the actual stride-rebase logic is still on
  the legacy path. Works correctly but leaves a small perf table on the
  floor when callers pass `overflow=True` unnecessarily.

---

## What lives where

```
op_tests/test_unified_attention_ck.py        # canonical test (grid + single-shape mode)
aiter/ops/unified_attention.py               # Python wrapper + transparent split-KV
aiter/jit/optCompilerConfig.json             # JIT build flags for module_unified_attention
3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/
    pipeline/unified_attention_pipeline.hpp  # core kernel (Tier-0/1/2 prefetch, refresh_*_offsets)
    pipeline/unified_attention_pipeline_default_policy.hpp  # policy / kBlockSize / LDS sizing
3rdparty/composable_kernel/example/ck_tile/42_unified_attention/
    unified_attention.cpp                    # select_config + KernelVariant dispatch
    unified_attention_impl.hpp               # per-variant compile-time knobs (kBlockPerCu, FP8 traits, ...)
ua-test-scripts/
    STATUS.md                                # this file
    test_single_shape.py                     # legacy specialised script (rocprof / GB/s reporting)
    regression_decode.sh                     # median-of-N decode sweep wrapper around test_single_shape.py
    probe_*.sh, tier2_*.sh, rocprof_analysis # historical investigation artifacts
```
