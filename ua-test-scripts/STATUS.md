# CK Unified Attention — Work-Log + Test Status

_Last updated: 2026-05-27. Status owner: jukorhon._

This document tracks the current state of the CK unified-attention work
that lives on the `jukorhon/unified-attention-ck` branch (same name on
both repos for coherence):

| Repo                          | Branch                              | HEAD (short) |
|-------------------------------|-------------------------------------|--------------|
| `ROCm/aiter`                  | `jukorhon/unified-attention-ck`     | bumped post-CK |
| `ROCm/composable_kernel`      | `jukorhon/unified-attention-ck`     | `46e622539`  |

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

### The one test script: `op_tests/test_unified_attention_ck.py`

Everything CK-UA-related goes through this. It has two modes:

* **Grid mode** (`--quick`, `--full`, no flags). Sweeps a Cartesian
  product of (seq_lens × num_heads × head_size × block_size × dtype ×
  q_dtype × num_blocks). Compares each row's CK output against the
  torch reference, and (off by default) optionally against the Triton
  kernel for a head-to-head perf number. Prints a DataFrame summary at
  the end. Exit-code-non-zero on any correctness failure → CI-ready.
* **Single-shape mode** (`-b -sq -sk` + dtype/head/block knobs).
  Expands to one batch of `b` identical `(sq, sk)` sequences, runs the
  same kernel path, and prints a focused report block (shape config,
  GPU, **split-KV status**, correctness vs reference, time, GB/s,
  TFLOPs, CK-vs-Triton speedup if applicable). The split-KV status row
  tags the report with `num_splits=N (ACTIVE|off)` so any perf surprise
  can be attributed to the combine-kernel + workspace cost vs. the
  attention kernel proper.

Triton comparison defaults are mode-aware: ON in single-shape (you
almost always want the head-to-head when investigating one shape), OFF
in grid runs (regression sweeps should be CK-vs-reference; Triton
numbers add runtime + decouple regression detection from Triton perf).
Override either default with `--triton` / `--no-triton`.

```bash
# Grid: default (~36 configs, a few minutes; CK vs ref only)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py

# Grid: smoke (~6 configs, <1 min)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py --quick

# Grid: full Triton-UA-style matrix (~216 configs, ~30 min)
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py --full

# Grid: include Triton head-to-head
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py --triton

# Single-shape (Triton on by default; `--num-blocks auto` sizes the
# KV pool so block_tables are unique per (seq, position) — no fake L2
# reuse, mirroring vLLM/SGLang allocator behaviour). `--dtype fp8`
# does the obvious thing: Q/K/V are quantised to e4m3.
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py \
    -b 64 -sq 1 -sk 128000 \
    --num-heads 64,8 --head-size 128 --block-size 16 \
    --dtype bf16 --num-blocks auto

# Single-shape, CK-only, FP8-quantised
HIP_VISIBLE_DEVICES=2 python op_tests/test_unified_attention_ck.py \
    -b 4 -sq 1 -sk 16384 \
    --num-heads 64,8 --head-size 128 --block-size 32 \
    --dtype fp8 --num-blocks auto --no-triton
```

### Sweep wrapper: `ua-test-scripts/regression_decode.sh`

Bash wrapper that runs the canonical script across the 4 (d, dtype)
decode combos at **two batch sizes**:

* `b=128` (light split-KV, `num_splits=2`) — measures the attention
  kernel itself; combine cost is negligible at this CTA count.
* `b=4` (heavy split-KV, `num_splits=64`) — measures the
  attention-kernel + combine pipeline. A regression that *only* shows
  up here lives in the combine path (Triton
  `reduce_segments_ck_layout`) or the wrapper's workspace alloc — not
  the CK attention kernel.

Each row is tagged with `splitkv=N` (and an `*` suffix when N > 1) so
the split-KV vs attention-only attribution is visible at a glance.
CK-only by default; pass `WITH_TRITON=1` to add Triton.

```bash
ua-test-scripts/regression_decode.sh                 # defaults: sk=16384, 5 runs, GPU 2, block=32
SK=32768 NUM_RUNS=7 GPU=3 ua-test-scripts/regression_decode.sh
WITH_TRITON=1 ua-test-scripts/regression_decode.sh   # add Triton comparison
```

---

## Recent changes on the branch

### CK side — `jukorhon/unified-attention-ck`

```
46e622539  CK-UA: gate dwordx3/x4 global_load_lds builtin on clang≥21, inline-asm fallback   [LATEST]
2645149bb  CK-UA: shrink Tier-2 page-table LDS cache to per-split window
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
<next>     unified_attention_ck: bump CK for clang<21 inline-asm fallback   [LATEST]
e78240f74  unified_attention_ck: bump CK + add decode regression script
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

## Toolchain portability: `global_load_lds_dwordx{3,4}` on ROCm ≤ 7.1.1

**Symptom (reported by a collaborator on a ROCm 7.1.1 container).**
JIT-building `module_unified_attention` failed in CK with

```
amd_buffer_addressing_builtins.hpp:2743:
  error: invalid size value
   __builtin_amdgcn_global_load_lds(gptr, lptr, 16, byte_offset_imm, kCoherence);
                                                ^~
note: size must be 1, 2, or 4
```

(and the same for `size=12`).

**Root cause.** AMD clang 20 (bundled with ROCm 7.1.1) only accepts `size`
∈ {1, 2, 4} for `__builtin_amdgcn_global_load_lds`. The `size=12` /
`size=16` ImmArg overloads for gfx950 only landed in clang ~21 (present
in ROCm 7.11.0 / clang 22; absent in ROCm 7.1.1 / clang 20). The literal
is checked during semantic analysis, so no compile flag avoids it. Our
CK pipeline always instantiates the `amd_async_global_load_lds_raw<…,
bytes={12,16}>` overload because the runtime `if(cache_ptr_int32_overflow_
possible)` dispatch in `unified_attention_pipeline.hpp` is *not* `if
constexpr`, so both branches are unconditionally compiled regardless of
the workload's cache size.

**Fix (CK `46e622539`).** Add a `CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN`
macro gated on `__clang_major__ >= 21`. When 1: existing `__builtin_…`
path. When 0: emit `global_load_lds_dwordx{1,3,4}` via inline asm, with
M0 set explicitly via `s_mov_b32` from the addrspace(3) `lptr` narrowed
to its 32-bit LDS byte offset and wave-uniformed via `readfirstlane`.
The inline asm bypasses the front-end ImmArg check; the assembler emits
the same HW instruction the builtin would have lowered to.

We tried two simpler fallbacks first and rejected both:

| Variant                            | b=128 / sk=16384 / d=128 / bf16 | b=1 / sk=1M / d=128 / bf16 |
|------------------------------------|---------------------------------|----------------------------|
| Builtin dwordx4 (baseline)         | 1.517 ms — PASS                 | 0.769 ms — PASS            |
| `N×` size=4 builtin, INST.OFFSET 0/4/8/12   | (skipped) — predicted wrong | 0.937 ms — **FAIL**        |
| `N×` size=4 builtin, INST.OFFSET 0/256/512/768 + `gptr` step  | 4.665 ms — **FAIL**             | 2.340 ms — PASS            |
| **Inline asm `dwordx4`** (chosen)  | **1.527 ms — PASS**             | **0.775 ms — PASS**        |

The decompositions only happen to produce a correct LDS layout for some
decode shapes — the in-LDS ordering of a native `dwordx4`'s 4 sub-issues
doesn't reduce to any combination of dword INST.OFFSET steps we could
find that survives all shapes. Asking the assembler for the literal
instruction sidesteps the question.

**Verified zero perf delta** on the full decode regression sweep (forced
the fallback on by temporarily flipping the cutoff to `__clang_major__
>= 999`); all 8 `(b, d, dtype)` configs match the builtin path to within
run-to-run noise (≤ 1.5%). The colleague can either upgrade to a newer
ROCm container (still the recommended option) or build the existing
`jukorhon/unified-attention-ck` branch on ROCm 7.1.1 unchanged — the
fallback activates automatically.

Override the heuristic manually if your toolchain straddles the cutoff:
`-DCK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN=0` to force the inline-asm
path, `=1` to force the builtin path.

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
op_tests/test_unified_attention_ck.py        # the test script (grid + single-shape mode)
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
    regression_decode.sh                     # median-of-N decode sweep across (b=128, b=4) × (d, dtype)
    rocprof_analysis/                        # local rocprof CSV dumps (gitignored)
```
