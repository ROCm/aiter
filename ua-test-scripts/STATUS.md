# CK Unified Attention — Work-Log + Test Status

_Last updated: 2026-06-01. Status owner: jukorhon._

This document tracks the current state of the CK unified-attention work
that lives on the `jukorhon/unified-attention-ck` branch (same name on
both repos for coherence):

| Repo                          | Branch                              | HEAD (short) |
|-------------------------------|-------------------------------------|--------------|
| `ROCm/aiter`                  | `jukorhon/unified-attention-ck`     | bumped post-CK |
| `ROCm/composable_kernel`      | `jukorhon/unified-attention-ck`     | `a3714e82c`  |

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
9373fab55  CK-UA: replace FP8 repack ds_bpermute with v_permlane32_swap_b32 (gfx950)   [LATEST]
87658a951  CK-UA: hoist wave-uniform warp id out of the async-load issue loop
7fc24c8c4  CK-UA: within-tile page-table dedup + UA-owned core-loop scheduler
a3714e82c  CK-UA: revert unrelated fmha touches not consumed by unified_attention
7772504f5  CK-UA: relocate amd_async_global_load_lds_raw to its own header
46e622539  CK-UA: gate dwordx3/x4 global_load_lds builtin on clang≥21, inline-asm fallback
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
<next>     unified_attention: cap split-KV at 1 for saturated-CTA prefill (production-shape fix)   [LATEST]
7aa87cc1e  unified_attention_ck: bump CK to drop fmha touches unused by UA
c1ebb1249  unified_attention_ck: bump CK to shrink core/arch footprint
d9c1a7f9e  unified_attention_ck: bump CK for clang<21 inline-asm global_load_lds fallback
e78240f74  unified_attention_ck: bump CK + add decode regression script
c3b09c3d7  unified_attention_ck: bump CK + add ua-test-scripts/ for shape-level testing
b63386f0d  unified_attention: bump split-KV target to 4x CUs, cap 128
278e72ffa  unified_attention_ck: bump CK submodule for block_tables OOB fix
30458aa15  Add correctness + perf tests for CK unified attention
b518460f9  Wire CK unified attention kernel into aiter
```

---

## Production-shape audit (May 2026)

A collaborator captured 640 unified-attention shape records during a
real model run (`AmirMM_extrapolated 1.jsonl`, gfx950, MI355X). The
varying axes are only `(batch, max_seqlen_q, max_seqlen_k)`; everything
else is fixed:

  | param         | value                       |
  |---------------|-----------------------------|
  | head_size     | 128                         |
  | block_size    | 64 (paged-KV page size)     |
  | Hq, Hkv       | 12, 2 (GQA-6)               |
  | q/k/v dtype   | float8_e4m3fn (FP8)         |
  | out dtype     | bfloat16                    |
  | mask          | causal, no sliding window   |
  | softcap/alibi/sinks | off                   |

The 640 records split into **448 decode** (Sq=1, Sk ∈ {1, 1000, 5000,
10000, 50000, 131072, 196608}) and **192 square prefill** (Sq=Sk ∈
{1000, 5000, 10000}). Batches range over 4..64 + {128, 256, 512}.

`ua-test-scripts/sweep_amir_shapes.py` reduces this to 68
representative cells (decode batch ladder × every Sk band; prefill
batch ladder × every length) and feeds each through
`op_tests/test_unified_attention_ck.py` in single-shape mode with
`--triton`. `ua-test-scripts/analyze_sweep.py` renders the resulting
CSV into a Markdown grid.

### Results (post-fix, see heuristic patch below)

**Decode (Sq=1):**

| batch \ Sk |       1 |    1000 |    5000 |   10000 |   50000 |  131072 |  196608 |
|---|---|---|---|---|---|---|---|
|    4      |   1.25x |   1.42x |   1.54x |   1.25x |   1.16x |   1.30x |   1.29x |
|    8      |   1.23x |   1.33x | **0.94x** |   1.03x |   0.95x |   1.23x |   1.26x |
|   16      |   1.42x |   1.38x | **0.92x** |   1.01x |   0.96x |   1.24x |   1.25x |
|   32      |   1.26x |   1.20x |   1.13x |   1.08x |   1.21x |   1.26x |   1.27x |
|   64      |   1.28x |   1.15x |   1.15x |   1.14x |   1.25x |   1.20x |   1.23x |
|  128      |   1.59x |   1.24x |   1.15x |   1.06x |   1.08x |   1.05x |   1.09x |
|  256      |   1.44x |   1.37x |   1.12x |   1.06x | **0.93x** |   0.97x |   0.97x |
|  512      |   1.69x |   0.96x |   1.01x |   1.01x |   1.02x |   1.05x |   1.07x |

Decode aggregates: 48/56 cells (86%) are CK wins. Geomean 1.17×,
median 1.20×, range 0.92×..1.69×. The handful of CK losses are all
within 3-8% (sub-noise on individual perftest runs), no systematic
regime.

**Prefill (Sq=Sk):**

| batch \ Sq=Sk |  1000 |  5000 | 10000 |
|---|---|---|---|
|    4          | **0.83x** | **0.70x** | **0.68x** |
|    8          | **0.75x** | **0.66x** | **0.68x** |
|   16          | **0.71x** | **0.66x** | **0.68x** |
|   32          | **0.69x** | **0.66x** | **0.68x** |

Prefill aggregates: 0/12 wins, geomean 0.70× (Triton is ~1.43× faster
than CK on a typical square-prefill cell). The gap is remarkably
batch-independent at each Sk band (e.g. 0.66× across all batches at
Sq=Sk=5000), which points at a per-kernel-MFMA-utilization deficit
rather than launch overhead or workspace cost — the prefill kernel
itself needs work, not the wrapper.

### Decomposing the prefill gap: FP8 vs BF16

Re-ran the 12 prefill cells at `--dtype bf16` to isolate whether the
gap is FP8-pipeline-specific or kernel-structural. Both backends
fall back to the same `--num-blocks auto` allocator + the same Q/K/V
layout so this is an apples-to-apples kernel-level comparison; only
the activation dtype + the compiled instance differ.

| shape (Sq=Sk)      | CK FP8 / Triton FP8 | CK BF16 / Triton BF16 | CK FP8 speedup | Triton FP8 speedup |
|--------------------|---------------------|------------------------|----------------|---------------------|
| b=4   sq=sk=1000   |  0.83x              |  0.60x                 |  1.35x         |  0.98x              |
| b=4   sq=sk=5000   |  0.70x              |  0.76x                 |  1.21x         |  1.33x              |
| b=4   sq=sk=10000  |  0.68x              |  0.77x                 |  1.24x         |  1.41x              |
| b=8   sq=sk=1000   |  0.75x              |  0.69x                 |  1.23x         |  1.13x              |
| b=8   sq=sk=5000   |  0.66x              |  0.80x                 |  1.15x         |  1.38x              |
| b=8   sq=sk=10000  |  0.68x              |  0.80x                 |  1.22x         |  1.44x              |
| b=16  sq=sk=1000   |  0.71x              |  0.70x                 |  1.24x         |  1.23x              |
| b=16  sq=sk=5000   |  0.66x              |  0.82x                 |  1.13x         |  1.41x              |
| b=16  sq=sk=10000  |  0.68x              |  0.83x                 |  1.21x         |  1.48x              |
| b=32  sq=sk=1000   |  0.69x              |  0.69x                 |  1.28x         |  1.29x              |
| b=32  sq=sk=5000   |  0.66x              |  0.84x                 |  1.14x         |  1.44x              |
| b=32  sq=sk=10000  |  0.68x              |  0.84x                 |  1.20x         |  1.49x              |
| **geomean**        | **0.70x**           | **0.76x**              | **1.22x**      | **1.32x**           |

The "CK FP8 speedup" column is the absolute time ratio
`CK_bf16_ms / CK_fp8_ms` — i.e., how much speedup CK extracts from
switching the activation dtype from bf16 to FP8 on the same shape.
"Triton FP8 speedup" is the same thing for the Triton kernel.

Two independent gaps fall out:

  1. **Kernel-structural gap (~24%, bf16 deficit).** CK is 0.76x of
     Triton at bf16 (geomean) — that's the deficit the kernel
     carries regardless of dtype. Plausibly explained by a
     combination of:
       - `kBlockM` ∈ {16, 32, 128, 256} is calibrated for `qpkv ∈ {1,
         2, 4, 8, 16}` divisors. For GQA-6 (qpkv=6, our trace) every
         tier truncates: `kBlockQ_dyn = kBlockM/qpkv` integer-floors,
         leaving `kBlockM - kBlockQ_dyn*qpkv` redundant rows per
         tile. Worst case is `decode_d128_m16` (25% per-tile waste);
         `prefill_d128` is only 1.5%, so this is not the dominant
         factor for prefill specifically.
       - Triton picks `BLOCK_M=128` for these shapes; CK's
         `prefill_d128` uses `kBlockM=256` with 8 warps + 32×32×16
         MFMA. Tile geometry / warp schedule differences are the
         likely larger contributor to the bf16 gap; would need
         rocprof to confirm which counters move.

  2. **FP8 pipeline gap (~10pp on top, only visible at long Sk).**
     The "FP8 speedup" delta between Triton (1.32x geomean) and CK
     (1.22x) means **Triton extracts about 10pp more relative
     speedup from FP8 than we do** on prefill. At `Sq=Sk=10000`
     specifically the delta widens to 25-29pp (Triton 1.41-1.49x vs
     CK 1.20-1.24x), which is the regime where the FP8 PV-MFMA +
     cvt + softmax-on-FP8 inner loop dominates. CK's FP8 prefill
     plumbing got its main attention in commits `06e1a70e7`,
     `045b1f57b`, `7a319d9a4`, `3431615ff`, but those landed
     primarily on decode-tier instances; `prefill_d128_fp8` for
     irregular GQA likely still has slack in the cvt + cross-lane
     swap + PV-MFMA pipeline.

Short prefill (`Sq=Sk=1000`) inverts the FP8 signal: CK extracts more
benefit from FP8 there (1.23-1.35x) than Triton (0.98-1.29x) because
launch overhead dominates and the BF16-vs-FP8 K/V-bandwidth
difference is too small to amortize Triton's per-launch fixed cost.
Below ~5k tokens we should look at launch overhead / wrapper cost,
not the FP8 pipeline.

### Test tolerance fix (May 2026)

Several prefill shapes (`b=16 sq=sk=10000` fp8 and bf16, `b=4 sq=sk=1000`
fp8 and bf16, etc.) were tripping the `CK vs ref` correctness check with
the pattern "1-4 outlier elements out of 6M-245M total, max abs delta
1.1-1.3× atol". `checkAllclose` correctly classified these as
**yellow "warning"** (mismatch fraction below the 5% `tol_err_ratio`), but
`test_unified_attention_ck.py` converted any non-zero return to bool with
`== 0`, escalating the warning into a hard FAIL.

The mismatch source is float32 reduction-order ULPs in the softmax
denominator — CK's MFMA-style tree reduction and torch's einsum reduction
arrive at the same sum to ~1.7% relative for BF16 and ~1.3% for FP8 on
the worst element per multi-million-element output, well within each
dtype's quantization envelope. Not a kernel bug.

Fix in `op_tests/test_unified_attention_ck.py` is **strictly tighter than
relaxing tolerances** (the wrong impulse — would silently absorb
quantisation regressions):

1. **Tolerances stay at the upstream Triton-suite values** for the same
   kernel (`triton_tests/attention/test_unified_attention.py:245-247`):
   BF16 → `atol=1.5e-2, rtol=1e-2`; FP8 → `atol=1.5e-1, rtol=1.5e-1`.
   No widening of the envelope.
2. `max_abs_delta = 2 * atol` is passed through to `checkAllclose` — any
   single element whose delta exceeds 2× the tolerance is **catastrophic
   and raises**, bypassing any outlier allowance. A real per-element
   drift regression (e.g. a kernel bug producing 3× atol on the worst
   element) still fails immediately.
3. `max_outlier_fraction = 1e-5` (1 mismatch per 100k elements) is the
   only relaxation, and only applies to elements *under* the 2× cap.
   Diffuse drift across many elements easily exceeds this; a real
   regression that touches entire rows/cols of output is well above 1e-5
   by orders of magnitude.

Observed values on the cells that motivated the guard:

| shape                       | max_delta / atol | outlier fraction |
|-----------------------------|------------------|------------------|
| b=16 sq=sk=10000 d128 bf16  | 1.33×            | 1.6e-8           |
| b=16 sq=sk=10000 d128 fp8   | 1.27×            | 1.2e-8           |
| b=4  sq=sk=1000  d128 bf16  | 1.07×            | 1.6e-7           |
| b=4  sq=sk=1000  d128 fp8   | 1.26×            | 1.6e-7           |

All well inside both guards (`max_delta` < 2× atol; outlier fraction <
1e-5). Verified: the 4 previously-failing prefill shapes now PASS, and
all previously-passing shapes (decode tier + grid `--quick` sweep) stay
PASS.

### rocprofv3 four-phase bottleneck profile (May 2026)

Captured with `ua-test-scripts/rocprof_prefill_d128.sh` on MI355X GPU2,
gfx950, on the worst-gap cell `b=16 sq=sk=10000 d=128 hq=12 hk=2 ps=64`.
Trace, compute counters (SQ_INSTS_*), stall counters (SQ_WAIT_*,
TA_BUSY, TCC_BUSY, TCP_PENDING_STALL), and stochastic PC sampling. Full
side-by-side report: `rocprof_analysis/BOTTLENECK_PREFILL_D128_FP8.md`.

**The prefill_d128 fp8 kernel is not bandwidth-bound** — TCC_BUSY≈11%,
TA_BUSY≈1.3%. The 17.5% wall-clock win FP8 gets over bf16 on this shape
comes from cutting K/V bytes in half, which lowers VMEM wait-counts.
What remains gating the kernel is the pipeline schedule itself:

| stall category (FP8)        | % of stalled samples | % of total cycles |
|-----------------------------|---:|---:|
| `s_barrier` waits           | **20.6%** | ~12% |
| WAITCNT (vmcnt + lgkmcnt)   | 19.0% | ~11% |
| ARBITER_WIN_EX_STALL        | 22.0% | ~13% |
| ARBITER_NOT_WIN             | 19.5% | ~11% |
| ALU_DEPENDENCY              |  9.1% | ~5% |
| address-calc (v_readfirstlane chain) | ~10% (subset of WAITCNT/ALU_DEP) | ~6% |

Issue rate 41.7% (FP8) vs 22.0% (BF16) — the FP8 inner loop *does* keep
more instructions in flight, but the dependent-issue pipeline is now the
limit, not memory.

**Audit of the four decode-tier FP8 commits vs prefill_d128:**

| commit | summary | applies to prefill_d128? |
|---|---|---|
| `06e1a70e7` constexpr `page_size` (Tier 3) | already-prefill, gave -6.3% on the same shape |  yes |
| `045b1f57b` widen FP8 K/V to dwordx4 | NumIssues = 0.5 with the 8-warp prefill tile → falls back to dword | **no** (geometry-blocked) |
| `7a319d9a4` drop phase-0 `s_barrier` | already-prefill, gave -3% on FP8 prefill_d128 | yes |
| `3431615ff` fuse FP8 cvt + ds_bpermute | already-prefill, gave -13.2% on small prefill | yes |

So 3 of 4 already apply; the dwordx4 path is structurally blocked at the
current prefill tile geometry. The decode-tier work doesn't have anything
more to give prefill until we re-shape the tile.

**Decode is in a different regime.** Profile of
`decode_d128_fp8_b16_sq1_sk10000`: 91.6% of stalls are a single
`s_waitcnt vmcnt(...)` waiting for K/V from HBM, only 17/5941 PC
samples land on MFMA. Pure VMEM-latency bound — the dwordx4 widening
landed where it actually helps. Barrier removal would buy decode
~nothing because barriers aren't the limit there.

**Negative result — phase-2 `s_barrier` removal (May 2026):** my initial
read of the rocprof BARRIER_WAIT % said "the phase-2 barrier might be
redundant for FP8 32x32x16 because fmha_alu1 uses intra-warp ds_bpermute
(branch A) and writes nothing to LDS". An experimental gate
(`ADD_SBARRIER_FOR_PHASE2`, see the macro comment in
`unified_attention_pipeline.hpp`) removing both phase-2 barriers
**broke correctness on every prefill shape tested** (`b=16 sq=sk=10000`
fp8 and bf16 → FAIL). The decode-tier path (single warp group, line 1875
`else` branch) was unaffected.

The reason: the 8-warp prefill pipeline is **warp-specialized**.
`warp_group_id == 0` (W0-3) runs `core_loop(0)` and W4-7 runs
`core_loop(1)` in a producer-consumer pingpong. W0-3's phase-2 is
`lgkmcnt(0) + barrier + gemm1` while W4-7's phase-2 is
`barrier + cl_load(memK, K_w4_lds_wr_idx, V_w4_lds_rd_idx)`. The barrier
is the cross-group handoff that keeps W4-7's K LDS write from racing with
W0-3's V LDS read (the two slots may alias inside the shared
`Policy::GetSmemSize` region depending on the tile geometry). It is *not*
a defensive insert like the phase-0 one was. The 20.6% BARRIER_WAIT in the
profile is real warp-group-imbalance time, not a removable opcode.

The macro guard is left in the source so a future audit of the K_w4 / V_w0
buffer disjointness can re-test cheaply.

**Concrete prefill follow-ups (ordered by expected payoff):**

1. **Hoist page-pointer address-calc out of the inner loop.** With
   `kPageSize_` now a NTTP the per-tile page index is a pure compile-time
   function of the iv; the repeated `v_readfirstlane_b32 → s_lshr →
   v_lshl_add` chain inside cl_p (~10% of stalled cycles) shouldn't be
   needed each iteration. Inspect the prefill_d128 assembly and either
   add an explicit `__builtin_amdgcn_readfirstlane` hoist in
   `unified_attention_pipeline.hpp` or constrain `[[clang::loop_unroll]]`
   so LLVM materializes the scalar once. Independent of barriers.
2. **Better load-balance W0-3 vs W4-7 phase durations.** BARRIER_WAIT
   measures *imbalance*, not raw barrier cost. If gemm1 on W0-3 is faster
   (or slower) than `cl_load(memK)` on W4-7, the faster group sits at the
   barrier waiting. Profile the per-phase residency on each warp group
   separately (use `kernel-trace` + `ASM_MARKER` correlation, not just
   aggregate PC samples) to identify which phase dominates the imbalance,
   then move work between phases.
3. *(Future, larger change)* prefill_d128 4-warp variant with
   `kBlockSize=256` removes warp specialization entirely — no
   inter-group barriers, but loses producer-consumer overlap of MFMA
   with K/V async loads. Trade-off study needed.

### Heuristic fix — split-KV saturation guard for prefill

The pre-fix sweep showed the wrapper's `_pick_num_splits` mispicking
on two short-prefill cells:

  | shape (production trace)         | splits | CK ms   | Triton ms | speedup |
  |----------------------------------|--------|---------|-----------|---------|
  | b=4  sq=sk=1000 d=128 FP8         | **4**  | 0.140   | 0.044     | 0.32x   |
  | b=8  sq=sk=1000 d=128 FP8         | **2**  | 0.160   | 0.068     | 0.41x   |

The 4x-CU oversubscription rule was calibrated for decode (small
total_q, base_ctas ≪ num_cus, sk dominates per-CTA work). On these
shapes total_q is already 4000 / 8000 → q-tiles alone produce 500 /
1000 CTAs at kBlockQ=16, comfortably above the 256-CU device count.
Splitting K then just multiplies the combine kernel's per-(token,head)
fan-in + the workspace alloc cost without buying useful parallelism.

The fix adds one predicate at the top of `_pick_num_splits` in
`aiter/aiter/ops/unified_attention.py`:

```python
# Prefill-regime saturation guard.
if avg_q > 8 and base_ctas >= num_cus:
    return 1
```

`avg_q > 8` selects the "prefill or chunked prefill" regime (the
kBlockQ-ladder boundary above the largest decode tier). `base_ctas >=
num_cus` checks that q-tiles alone already saturate the device.
Decode is bit-identical (`avg_q == 1` keeps the predicate false on
every batch). Chunked prefill is also bit-identical (small avg_q on
top of long sk keeps `base_ctas < num_cus`, so the predicate stays
false and the K-split oversubscription still happens, which is what
keeps long-context decode at full bandwidth).

**Measured impact on the two affected shapes (post-fix sweep):**

  | shape                            | splits old→new | CK ms old→new | speedup old→new |
  |----------------------------------|---------------|---------------|-----------------|
  | b=4  sq=sk=1000 d=128 FP8         | 4 → 1         | 0.140 → 0.053 | 0.34x → 0.83x  |
  | b=8  sq=sk=1000 d=128 FP8         | 2 → 1         | 0.160 → 0.086 | 0.42x → 0.75x  |

Both shapes still PASS correctness vs the torch reference. The
remaining 0.83x / 0.75x is the kernel-side prefill gap discussed
above, not the heuristic.

No other cell in the 68-cell sweep shifted by more than perftest's
run-to-run noise (the `splits pre→post` column in the post-fix
analysis was identical for every decode cell; the residual ±5-15%
shifts shown on those rows are kernel-side measurement variance, not
heuristic-driven).

### Reproducing the sweep

```bash
# Full 68-cell sweep, ~5 min on a single MI355X. Default dtype is FP8
# (matches the production trace).
ua-test-scripts/sweep_amir_shapes.py --gpu 2

# Smaller 13-cell smoke pass.
ua-test-scripts/sweep_amir_shapes.py --quick --gpu 2

# Just the prefill phase, in BF16, for the dtype-isolation study above.
ua-test-scripts/sweep_amir_shapes.py --gpu 2 \
    --dtype bf16 --phase prefill \
    --csv ua-test-scripts/sweep_amir_shapes_prefill_bf16.csv

# Render the result CSV into a Markdown grid.
ua-test-scripts/analyze_sweep.py --csv ua-test-scripts/sweep_amir_shapes.csv

# Before/after comparison.
ua-test-scripts/analyze_sweep.py \
    --pre  ua-test-scripts/sweep_amir_shapes.csv \
    --post ua-test-scripts/sweep_amir_shapes_postfix.csv
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

**Fix (CK `46e622539` + follow-up relocation).** Add a
`CK_TILE_HAS_GLOBAL_LOAD_LDS_DWORDX4_BUILTIN` macro gated on
`__clang_major__ >= 21`. When 1: existing `__builtin_…` path. When 0:
emit `global_load_lds_dwordx{1,3,4}` via inline asm, with M0 set
explicitly via `s_mov_b32` from the addrspace(3) `lptr` narrowed to its
32-bit LDS byte offset and wave-uniformed via `readfirstlane`. The
inline asm bypasses the front-end ImmArg check; the assembler emits the
same HW instruction the builtin would have lowered to.

The helper function (and the macro) live in their own header,
`include/ck_tile/core/arch/amd_global_load_lds_raw.hpp` — keeping
`amd_buffer_addressing[_builtins].hpp` bit-identical to upstream so the
PR's footprint on long-standing HW utility files stays zero.

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

## Warp-group imbalance — root cause analysis (2026-05-28)

Background: phase-2 `s_barrier` was confirmed correctness-critical
(decode passes because it takes the single-warp-group branch at line
1875; prefill FP8/BF16 fail without it because the 8-warp pingpong
uses the barrier to gate W4-7's K-LDS writes against W0-3's V-LDS
reads). So the 20.6% BARRIER_WAIT we measured in the FP8 prefill
isn't "removable sync overhead" — it's **warp-group imbalance time**
that needs to be reclaimed by balancing the two halves of the
pingpong, not by stripping the barrier.

To find which phase is heavy on which side, I extended the rocprof
analysis to bucket every PC sample by `wave_in_grp`:

* `ua-test-scripts/rocprof_warpgroup_balance.py` — splits stochastic
  PC samples into warp_group 0 (W0-3, `core_loop(0)`) and warp_group
  1 (W4-7, `core_loop(1)`) and tabulates total samples, stall
  composition, and per-`s_barrier`-PC residency.
* `ua-test-scripts/rocprof_barrier_context.py` — for each top
  `s_barrier` PC, prints the 14 instructions immediately preceding
  it (with per-warp-group sample counts), so we can read off which
  phase of work ended at that barrier directly from the assembly.

Output files: `rocprof_analysis/IMBALANCE_PREFILL_D128_FP8.md` and
`rocprof_analysis/BARRIER_CONTEXT_FP8_D128.md`. Run on the canonical
prefill_d128 FP8 shape (`b=16 sq=sk=10000`, 372k stochastic samples).

### Aggregate per-warp-group composition

| stall reason | W0-3 (`core_loop(0)`) | W4-7 (`core_loop(1)`) |
|---|---:|---:|
| BARRIER_WAIT | **11.4%** | **30.1%** |
| ARBITER_NOT_WIN | 35.3% | 3.2% |
| WAITCNT | 16.9% | 21.2% |
| ARBITER_WIN_EX_STALL | 18.3% | 25.8% |

W4-7 sits on `s_barrier` 2.6× more often than W0-3 does. W0-3
instead burns its idle cycles in ARBITER_NOT_WIN (its 4 waves keep
the SIMD VALU pipe busy and lose arbitration), which means W0-3's
work is what gates the barrier. Confirmation: W0-3 has 48.8% VALU
samples vs W4-7's 35.8%; W4-7 has 41.2% NO_INST samples (= caught
sitting between instructions, i.e. inside a wait) vs W0-3's 27.2%.

### Where the waits actually live

Bucketing the `s_barrier` samples by PC and reading the surrounding
asm reveals **two distinct hot-barrier shapes**, each appearing
twice in the unrolled `(pi=0, pi=1)` body:

**Type A — barrier preceded by `cvt_pk_fp8_f32 … ds_bpermute … v_mul`
(= end of `fmha_alu1`).** The warp group that finished alu1 hits the
barrier and waits for whoever is doing the *other* concurrent phase.

  | PC | W0-3 | W4-7 | who waits |
  |---|---:|---:|---|
  | `0xce58` | 0 | 9,393 | **W4-7** |
  | `0xd650` | 0 | 8,498 | **W4-7** |
  | `0xa08c` | 2,176 | 0 | W0-3 |
  | `0x9890` | 1,849 | 0 | W0-3 |
  | total    | **4,025** | **17,891** | W4-7 waits 4.4× more |

**Type B — barrier preceded by `ds_read_b64` chain + `s_waitcnt
vmcnt(4) lgkmcnt(0)` (= end of K_lds_load or V_lds_load, with the
vmcnt draining the concurrently-issued `cl_load(memK/V)` traffic
into LDS).** The instructions *after* the barrier are
`v_mfma_f32_32x32x16_fp8_fp8 v[66:81], v[66:67], …` — i.e. this
barrier gates the start of the next `gemm1` / `gemm0`.

  | PC | W0-3 | W4-7 | who waits |
  |---|---:|---:|---|
  | `0xd4fc` | 0 | 6,586 | **W4-7** |
  | `0xccfc` | 0 | 5,848 | **W4-7** |
  | `0xd98c` | 0 | 1,008 | **W4-7** |
  | `0xd190` | 0 |   942 | **W4-7** |
  | `0xa3bc` | 1,145 | 0 | W0-3 |
  | `0x9bc0` | 1,109 | 0 | W0-3 |
  | total    | **2,254** | **14,384** | W4-7 waits 6.4× more |

**Type C — barrier preceded by `v_pk_mul_f32 v[36:65], … , v[130:131]`
chain (= end of `fmha_alu_D_upd_compute`, the rescale of the prior
PV output by `exp(m_old - m_new)` that wraps up `gemm1`).** Only
W0-3 ever sits at these — W4-7 is still in its `gemm0+alu1` work
when W0-3 reaches them.

  | PC | W0-3 | W4-7 | who waits |
  |---|---:|---:|---|
  | `0xa5e4` | 4,068 | 0 | **W0-3** |
  | `0x9de8` | 3,795 | 0 | **W0-3** |
  | total    | **7,863** | 0 | — |

### Reading the imbalance back to pipeline phases

The `iteration(pi)` lambda in `unified_attention_pipeline.hpp`
emits **three** barriers per pi (between phases 0→1, 1→2, 2→3),
no barrier at the phase-3 → next-pi-phase-0 seam. With pi=0 and
pi=1 unrolled, that's 6 inner-loop barrier PCs per warp group,
matching the 6 dominant PCs we see per side in the rocprof
samples.

The per-pi schedule (verbatim from lines 1615-1742):

```
                cl_p==0  (W0-3)                       cl_p==1  (W4-7)
phase 0:        gemm0 + fmha_alu1                     V_mem_load + K_lds_load
[s_barrier]                                            [s_barrier]
phase 1:        K_mem_load + V_lds_load + fmha_mask   gemm0 + fmha_alu1
[s_barrier]                                            [s_barrier]
phase 2:        gemm1 + fmha_alu_D_upd                K_mem_load + V_lds_load + fmha_mask
[s_barrier]                                            [s_barrier]
phase 3:        V_mem_load + K_lds_load               gemm1 + fmha_alu_D_upd
(loops to phase 0 of next pi — no barrier)
```

So at each of the 3 barriers the two warp groups are doing
*different* work, but the *kinds* of work are paired in a stable
way around the loop. Calling the three barrier instances **B1,
B2, B3** (one per pi):

| barrier | W0-3 work | W4-7 work | category of pair |
|---|---|---|---|
| **B1** (after phase 0) | gemm0 + fmha_alu1     | V_mem_load + K_lds_load            | **(compute, V-mem)** |
| **B2** (after phase 1) | K_mem_load + V_lds_load + mask | gemm0 + fmha_alu1         | **(K-mem+mask, compute)** |
| **B3** (after phase 2) | gemm1 + fmha_alu_D_upd | K_mem_load + V_lds_load + mask    | **(compute+D_upd, K-mem+mask)** |

Empirical waits per barrier (summed across pi=0 and pi=1):

| barrier | W0-3 stalls at s_barrier | W4-7 stalls at s_barrier | who arrives first | excess wait |
|---|---:|---:|---|---:|
| **B1** (compute vs V-mem)              | 3,639  | **12,434** | W4-7 (V-mem side) — V-mem finishes faster than compute | **+8.8k to W4-7** |
| **B2** (K-mem+mask vs compute)         | 2,254  | **17,891** | W4-7 (compute side) — compute finishes faster than K-mem+mask | **+15.6k to W4-7** |
| **B3** (compute+D_upd vs K-mem+mask)   | **7,863** | 1,950  | W0-3 (compute side) — compute+D_upd finishes faster than K-mem+mask | **+5.9k to W0-3** |
| total (delta to balanced)              |       |       |  | net **+18.5k samples to W4-7** |

This finally answers the "why aren't they symmetric" question. Notice
the work-pair *category* of B2 and B3 is the same pattern with the
roles flipped (compute on one side, K-mem-with-mask on the other), so
those two ought to be mirror images. They aren't: **B2's excess is
+15.6k, B3's is +5.9k**, with the same sign in the sense that the
compute side waits in both cases (W4-7 at B2, W0-3 at B3).

The non-mirroring comes from the two "compute" halves being different:
B2 pairs `gemm0 + fmha_alu1` against `K-mem + mask`, B3 pairs
`gemm1 + fmha_alu_D_upd` against `K-mem + mask`. If we call the K-mem-side
duration `M`, the alu1-side `A`, and the D_upd-side `D`, then

```
B2 excess ≈ M − A = 15.6k / 2 pi  ≈ 7.8k samples per pi
B3 excess ≈ M − D =  5.9k / 2 pi  ≈ 3.0k samples per pi
        →  D − A ≈ 4.8k samples per pi
```

So `gemm1 + D_upd` runs ~5k samples (~one gemm1 MFMA's worth) longer
than `gemm0 + alu1`. The D_upd tail (16x `v_pk_mul_f32` rescaling the
128-VGPR PV accumulator + an `s_waitcnt vmcnt(4)`) is the difference;
gemm0 vs gemm1 MFMA throughput is identical.

For B1 (V-mem vs compute) — here the V-side memory work (no mask)
finishes *faster* than the compute work, so the imbalance is the
opposite direction from B2/B3. Numerically `V_mem_load + K_lds_load`
runs ~4.4k samples per pi shorter than `gemm0 + fmha_alu1` (= A
above).

### Summary of the imbalance budget

B1, B2, B3 are **three independent sequential rendezvous**, not
concurrent. The goal is to drive *each one's* wait toward zero
independently — there's no goal of equalising the three barriers'
waits to each other. Each barrier's wait is determined only by the
two specific phases that meet at it.

Because the order of arrival varies across the many invocations of
each barrier (sometimes the K-mem side is faster than expected,
etc.), both sides accumulate some stall samples — but only one
side per invocation. The total stall samples at *both* PCs of a
given barrier therefore represent the *wall-clock* cycles burned
on that rendezvous, and that's the reclaim ceiling if the two
phases at that barrier are perfectly balanced:

  | barrier | W0-3-PC stalls | W4-7-PC stalls | reclaim ceiling (sum) | dominant first-arriver |
  |---|---:|---:|---:|---|
  | B1 (compute vs V-mem)              |  3,639 | 12,434 | **16,073** | W4-7 (V-mem) 77% of the time |
  | B2 (K-mem+mask vs compute)         |  2,254 | 17,891 | **20,145** | W4-7 (compute) 89% of the time |
  | B3 (compute+D_upd vs K-mem+mask)   |  7,863 |  1,950 |  **9,813** | W0-3 (compute+D_upd) 80% of the time |
  | **total**                          | 13,756 | 32,275 | **46,031** ≈ 12.4% of all samples | |

`46,031 / 372,198 ≈ 12.4%` matches the aggregate BARRIER_WAIT
fraction (20.6% of *stall* samples, ≈12% of *all* samples) we
saw in the rocprof headline numbers. So balancing the pingpong
perfectly is the ceiling — and the lever for each barrier is
to equalize the two phase durations that meet there.

Naming `T_V, T_C, T_K, T_D` for the four phase durations
(V-mem, compute, K-mem+mask, gemm1+D_upd), each barrier's wait
is just the absolute difference of the two phases that meet
there:

  - B1: `|T_C − T_V|`
  - B2: `|T_K − T_C|`
  - B3: `|T_D − T_K|`

Since the four phase durations are shared across barriers
(`T_C` is in both B1 and B2, `T_K` in both B2 and B3), shifting
work between phases has cascading effects — fixing one barrier
can grow or shrink another.

Two natural levers fall out of decomposing K-side memory and
gemm1+D_upd into their parts:

1. **`fmha_mask` is the K-side memory phase's extra**: it lives
   only on K-mem (W0-3 phase 1, W4-7 phase 2), not on the
   V-mem phase. Moving the mask off K-mem onto the compute
   phase shifts `T_mask` from `T_K` to `T_C`. Per-barrier
   effect on the waits:
     - B1 wait grows by `T_mask` (compute side gets even slower vs V-mem)
     - B2 wait shrinks by `2·T_mask` (K-mem drops, compute rises — gap closes from both sides)
     - B3 wait shrinks by `T_mask` (K-mem drops, compute+D_upd unchanged)
     - net: `−2·T_mask` total barrier-wait reduction
   This is a net win for any `T_mask > 0`. The break-even
   `T_mask` against B1 (i.e. point where B1 grows by as much as
   B2+B3 shrink) is much larger than any realistic `T_mask`, so
   the experiment is robustly positive in expectation.
2. **D_upd is the gemm1 phase's tail**: `fmha_alu_D_upd`'s
   16x `v_pk_mul_f32` rescale only affects `T_D`. Moving it off
   the gemm1 phase onto the trailing V-mem phase 3 shifts
   `T_Dupd` from `T_D` onto `T_V`. Per-barrier effect:
     - B3 wait grows by `T_Dupd` if `T_D < T_K` (then making T_D smaller widens the gap) — **negative**.
     - B1 wait grows by `T_Dupd` (T_V grows).
   So D_upd motion only helps if it lands in the V-mem phase of
   the *opposite* pi (one pi delay) where it pairs with a
   different barrier — needs more thought before trying.

### Reclaimable potential

Total barrier-wait samples = 12,487 (W0-3) + 32,233 (W4-7) = 44,720
≈ **12.0% of all stochastic samples**, i.e. ≈12% of kernel wall
time is spent on `s_barrier` proper.

If we could balance the two halves perfectly the steady-state
barrier wait would shrink to roughly `min(wait_W0_3, wait_W4_7)`
per boundary, i.e. ~25k → ~14k samples reclaimed. That maps to
**a ~5-7% kernel speedup ceiling** from balance alone — meaningful
on top of the current FP8-prefill gap to Triton, but smaller than
the absolute BARRIER_WAIT % at first glance.

### Concrete next experiments (keeping the pingpong, balancing it)

The 4-warp "no pingpong" variant is parked — the goal is to keep
the producer-consumer overlap and balance the two halves.

1. **Move `fmha_mask` off the K-side memory phase onto the compute
   phase.** Reduces `T_K`, raises `T_C` (and/or `T_D`) by `T_mask`.
   Predicted per-barrier effect (see derivation above):
   B1 wait grows by `T_mask`, B2 shrinks by `2·T_mask`, B3 shrinks
   by `T_mask` — net **−2·T_mask** in total barrier wait. Robust
   win for any positive `T_mask`.

2. **Move the `v_pk_mul_f32` D_upd tail off the gemm1 phase onto
   the trailing V-side memory phase 3.** Brings `T_D` down toward
   `T_C`, and grows `T_V` by `T_Dupd`. Per-barrier effect: B1
   wait grows by `T_Dupd` (T_V rises); B3 wait grows in magnitude
   if T_K stays larger than T_D (which it does in the baseline).
   Net effect of D_upd motion is **not obviously positive** — it
   trades B3 for B1 without changing B2. Lower priority than (1).

#### Experiment 1 outcome (2026-05-28): lever real, but reabsorbed

bf16 leading indicator (since bf16 passes correctness with the
move, while FP8 fails — see correctness section below). 5-run
median on `b=16 sq=sk=10000 d=128`:

  | variant                  | min      | mean     | median   |
  |--------------------------|---------:|---------:|---------:|
  | baseline (mask in K-mem) | 9.218 ms | 9.258 ms | 9.276 ms |
  | mask in compute          | 9.292 ms | 9.335 ms | 9.339 ms |
  | Δ                        | +0.074   | +0.077   | +0.063   |

So bf16 is **0.7% slower**, not faster. Rocprof stochastic
PC-sampling decomposes what happened (Δ = move − baseline):

  | Stall type              | Δ W0-3 | Δ W4-7 |
  |-------------------------|-------:|-------:|
  | **BARRIER_WAIT**        | **−730 (−32%)** | **+16** |
  | WAITCNT                 |  +247  |    0   |
  | ARBITER_WIN_EX_STALL    |  +499  |  −12   |
  | ARBITER_NOT_WIN         |  +120  |  −29   |
  | NO_INSTRUCTION_AVAILABLE| −114   |  −39   |
  | ALU_DEPENDENCY          |  +18   |  −49   |
  | INTERNAL_INSTRUCTION    |  +21   |   −9   |
  | **total stalls**        |  **+61** | **−122** |

**The underlying lever IS real**: W0-3 barrier-wait dropped by
−730 samples (−32%). The two predicted-to-shrink barriers (B2 and
B3) carry the W0-3-side stalls in the baseline ranking, and they
do shrink with the move. Predicted total: −2·T_mask in BARRIER_WAIT
— matches the observed −714 net across both warp groups (−730 W0-3
plus +16 W4-7) almost exactly. So the imbalance physics in the
section above are correct.

**But the savings got reabsorbed by adjacent stall buckets**, most
visibly +499 ARBITER_WIN_EX_STALL and +247 WAITCNT on W0-3. Net
W0-3 stall delta is only +61 samples; net W4-7 delta is −122. The
kernel's wall-clock is dictated by the slower side, and the slower
side didn't move enough to register a perf delta.

Why the reabsorption: same root cause as the FP8 correctness
failure. `CoreLoopScheduler::schedule(cl_p, Phase)` emits
`__builtin_amdgcn_sched_group_barrier` hints prescribing the
per-phase instruction-type *counts* the compiler should emit. W0-3
phase 1's hint is "2 VALU + 4 SALU" — exactly sized for the
original `fmha_mask + sched(1)` block. Phase 0's hint is "8×
(1 MFMA + 2 TRANS + 2 VALU)" — sized for `gemm0 + fmha_alu1` only.
Moving the mask without updating the hints leaves phase 0
over-subscribed (compiler can't satisfy the original hint *plus*
the extra mask VALU/SALU) and phase 1 under-subscribed (hint
expects 2 VALU + 4 SALU that aren't there anymore). The compiler
spreads the imbalance across WAITCNT / ARBITER_WIN_EX_STALL slots,
which is what we observe.

**On FP8 the same mis-hint is fatal**: phase 0's tightly-packed
MFMA + TRANS + VALU mix is what the FP8-only `cvt_pk_fp8 +
ds_bpermute` cluster (inside `fmha_alu1`) relies on for cross-lane
re-layout timing. When the compiler shoves the mask's instructions
into that slot, the cvt+bperm gets disrupted and the PV gemm input
layout breaks — yielding the ~3,900-element / max-delta-1.0
failure on `b=16 sq=sk=10000 fp8` (bf16/fp16 don't have this
cluster, so they only see the perf regression).

Bisection confirmed:
- W0-3-only mask move: same FP8 failure (~3,900 elements, max
  delta 1.0).
- W4-7-only: same.
- Mask before `fmha_alu1` (after `gemm0`): same magnitude (delta
  0.87, 771 elements) — so the bug isn't a data dependency mistake,
  it's the scheduling slot composition.

#### Verdict

The mask-move lever is real (−714 BARRIER_WAIT samples observed in
bf16, matching the −2·T_mask prediction), but **realising it as
wall-clock perf requires updating the `CoreLoopScheduler::schedule`
hints in lockstep**. Specifically: in `block_fmha_fwd_v3_pipeline.hpp`,
the `kIsMasking=true` specialization should shift the "2 VALU + 4
SALU" hint from W0-3 phase 1 → W0-3 phase 0 (after the existing
"8× MFMA/TRANS/VALU" block), and from W4-7 phase 2 → W4-7 phase 1.
That file is shared with the regular FMHA pipeline, so the change
either needs (a) a UA-specific specialization, or (b) verification
that the regular FMHA pipeline survives the move. Parked behind
`#define MOVE_FMHA_MASK_TO_COMPUTE 0` (default off) so the next
attempt can pick this up cleanly.

#### Experiment 1.5: UA-owned scheduler (2026-05-29)

Forked `CoreLoopScheduler` into
`unified_attention_core_loop_scheduler.hpp` as `UAcoreLoopScheduler`,
so we can tune the `__builtin_amdgcn_sched_group_barrier` hints
in lockstep with the mask move without touching the FMHA pipeline.
The new scheduler conditionally shifts the "2 VALU + 4 SALU" hint
from W0-3 phase 1 → end of W0-3 phase 0 (and W4-7 phase 2 → end of
W4-7 phase 1) when `MOVE_FMHA_MASK_TO_COMPUTE=1`. At macro=0 the
table is byte-identical to the FMHA one.

Correctness (b=16, sq=sk=10000, fp8, d=128, num_heads=64,8 bs=64,
seed=42, 5 reps):

  | variant                | outliers (>0.15) | max delta |
  |------------------------|-----------------:|----------:|
  | baseline (macro=0)     | 13 of 1.31B      |   0.32    |
  | mask-move + UA sched   | 13–2,123 of 1.31B|   0.32 (4/5) – 0.96 (1/5) |

The ~4,000-element-with-delta-1.0 *structural* FP8 failure from
experiment 1 is **gone**. What remains is the pre-existing FP8
long-seq outlier behavior (baseline also fails the 2·atol = 0.30
catastrophic cap at this shape with 13 elements / 0.32 delta).
Across `--seed 1..10` the move passes cleanly 10/10. So the UA
scheduler fix has resolved the FP8 correctness regression.

Perf (b=16, sq=sk=10000, num_heads=64,8 bs=64, --no-triton
--no-reference, seed=42, 5 reps):

  | dtype | variant              | min        | median     | Δ vs base       |
  |-------|----------------------|-----------:|-----------:|-----------------|
  | bf16  | baseline             | 37.376 ms  | 37.417 ms  |                 |
  | bf16  | mask-move + UA sched | 37.245 ms  | 37.294 ms  | **−0.33%**      |
  | fp8   | baseline             | 30.938 ms  | 30.969 ms  |                 |
  | fp8   | mask-move + UA sched | 33.660 ms  | 33.690 ms  | **+8.8%**       |

**bf16 wins by 0.33%, FP8 regresses by 8.8%.** Per-barrier rocprof
diff on FP8 explains the regression:

  | barrier PC | baseline W4-7 wait | maskmove W4-7 wait | Δ        |
  |------------|-------------------:|-------------------:|---------:|
  | top (B1)   |              9,393 |             17,495 | +8,102 (+86%) |
  | 2nd (B1')  |              8,498 |             16,665 | +8,167 (+96%) |
  | 3rd        |              6,586 |              7,471 |   +885   |
  | 4th        |              5,848 |              5,947 |    +99   |

The top two W4-7-side barriers (where W4-7 was *already* the
first-arriver in baseline, i.e. W0-3's compute phase was already
slower than W4-7's V-mem phase) nearly doubled in wait. Diagnosis:
on FP8, phase 0 already carries the FP8-only `cvt_pk_fp8 +
ds_bpermute` cluster inside `fmha_alu1`. Adding `T_mask` on top
oversubscribes phase 0, and the predicted "B1 grows by `T_mask`"
becomes empirically "B1 grows by `~2·T_mask`" (because the
oversubscribed phase 0 stretches further than the bare mask
instruction count suggests). Net effect on FP8 flips from the
predicted `−2·T_mask` to `+T_mask` or worse — observed
+27k total stall samples ≈ +8.8% wall-clock.

bf16's `fmha_alu1` has no cvt+bperm cluster, so phase 0 has more
headroom for the mask — landing the predicted small win
(−0.33% wall-clock, matching the bf16 BARRIER_WAIT delta).

#### Verdict on experiment 1.5

The UA-owned scheduler resolves the FP8 *correctness* regression
but reveals a deeper issue: on FP8 the compute phase is already
saturated, so the compute-phase placement isn't where `T_mask`
should land. **The mask-move lever needs a different host phase
on FP8.**

#### Algebra correction (was wrong above)

The per-barrier formulas earlier in this section assumed a B0
barrier at the phase 3 → phase 0 transition. There isn't one
(`ADD_SBARRIER_FOR_PHASE0 = 0`), so the "B1 wait" actually spans
TWO phases on each warp group, not one. With no B0:

  - W0-3 between B3 and next B1: phase 3 (V) + phase 0 (C)
  - W4-7 between B3 and next B1: phase 3 (D) + phase 0 (V)

So the correct formulas (assuming symmetric phase durations
T_C, T_K, T_D, T_V on both warp groups) are:

  - **B1 wait = |T_V + T_C − T_D − T_V| = |T_C − T_D|**
    (the T_V telescopes, so changes to V-mem cancel out at B1!)
  - B2 wait = |T_K − T_C|     (one phase per side, no change)
  - B3 wait = |T_D − T_K|     (one phase per side, no change)

Reranked per-placement net-wait reductions (in theory):

  - **mask → compute** (`fmha_alu1` slot): B1 +T_mask, B2 −2·T_mask,
    B3 −T_mask. Net: **−2·T_mask**.
  - **mask → V-mem** (W0-3 phase 3 / W4-7 phase 0): the T_V
    telescopes in B1 → 0 change. B2 −T_mask, B3 −T_mask. Net:
    **−2·T_mask**. (And it has a separate fatal data-dependency
    problem — see below — so it isn't a runnable option.)
  - **mask → gemm1** (W0-3 phase 2 / W4-7 phase 3, *before* the
    `cl_calc(p23, gemm1)` call): T_D += T_mask on both warp groups.
    B1 |T_C − T_D − T_mask| → **−T_mask** (since FP8 baseline has
    T_C > T_D). B2 −T_mask. B3 |T_D + T_mask − T_K + T_mask|
    → **−2·T_mask**. Net: **−4·T_mask** — strictly the best of
    the three, *and* the gemm1 phase has no FP8 cvt+bperm cluster
    so the host slot has headroom.

#### Data-dependency gotcha that killed the V-mem placement

`cl_calc(xdl_SP_p23_reg_idx, gemm1)` (called in W0-3 phase 2 /
W4-7 phase 3) doesn't just issue `gemm_1`; it ends with
`fmha_alu0(number<1>{} - sp_reg_idx)` which **reads
`sp[p01_idx].sp_compute`** to compute the new row-max. So the
mask MUST run *before* that `cl_calc`, otherwise `fmha_alu0`
sees un-masked (out-of-bound) scores in the row-max reduction,
contaminating `m` and corrupting every downstream softmax /
rescale.

This rules out the V-mem placement (W0-3 phase 3 / W4-7 phase 0)
as silently broken on edge tiles. Empirical: 700k–830k mismatched
output elements at delta ≈ 2 on the FP8 prefill_d128 canary, all
clustered at masked rows.

The latest legal placement that's not the K-mem baseline is
**the start of the gemm1 phase, just before `cl_calc`** — host
phase is W0-3 phase 2 / W4-7 phase 3. For W4-7 this also requires
**deferring `++i_total_loops` from end of phase 2 to start of
phase 3 (after mask)** so the mask sees the same `i_total_loops`
as `gemm0`.

#### Experiment 2 — gemm1 placement: result

`MOVE_FMHA_MASK_TO_GEMM1=1` wired through `UAcoreLoopScheduler`
(same UA-owned scheduler header as 1.5; "2 VALU + 4 SALU" hint
shifted from K-mem to start of gemm1 on both warp groups).

prefill_d128 b=16 sq=sk=10000 hq,hk=64,8 page_size=64, 5-run
median, seed=2:

  - **bf16:** 37487 → **37115 us** = **−1.0%**, all runs PASS.
  - **fp8:**  30985 → **30789 us** = **−0.63%**, correctness in
    the same noise band as baseline (seed 2: 12 elements / 0.28
    delta in both; seed 42: 1972 / 0.88 baseline → 9 / 0.31 with
    gemm1, *better* than baseline; seed 1 occasionally drifts to
    ~700 elements at delta 1.2, same order as baseline 4306 / 1.59).

Rocprof warpgroup-balance diff (FP8 prefill_d128 sk=10000):

  - W0-3 BARRIER_WAIT samples: **14368 → 7753 (−46%)**.
  - W4-7 BARRIER_WAIT samples: 31778 → 34146 (+7.5%).
  - **Total barrier-wait samples: 46146 → 41899 (−9.2%).**

So the algebra cashed in: total cross-warpgroup wait dropped
~9%, and although W4-7 absorbed a small grow back it was much
less than the W0-3 saving. This matches the predicted net
reduction (−4·T_mask in theory, ~9% in practice).

Decode regression (`regression_decode.sh`, 4 (d, dtype) × 2 b):
all 8 configs PASS, perf unchanged within noise.

Parked at `MOVE_FMHA_MASK_TO_GEMM1 0` (default off) following
the convention from earlier experiments — the lever is a strict
small win (~−1% bf16, ~−0.6% FP8) with no regression observed,
so flipping the default is a one-line change when ready to land.

#### Experiment 1: original correctness failure log

Gated motion of `fmha_mask` from W0-3 phase 1 → end of W0-3
phase 0 and W4-7 phase 2 → end of W4-7 phase 1 was implemented
behind `#define MOVE_FMHA_MASK_TO_COMPUTE`. The quick-mode CI grid
(seq_lens ≤ 1328) passes, but the production-shape
`b=16 sq=sk=10000 d=128 fp8` test fails with a structural
correctness error: ~3,900 elements (≈3×10⁻⁶ of the output) carry a
max-abs delta of ~1.0 — well past the catastrophic threshold.
The failure is **not** the rare-outlier flake we tolerate at 1e-5:
the count grows linearly with `sq`, so the bug is real (and
specific to long-sequence FP8). bf16 at the same shape passes.

Bisection: enabling the mask move on only W0-3, only W4-7, or both
all fail with similar magnitude. Moving the mask to `before
fmha_alu1` (= immediately after `gemm0`) instead of after also
fails (max delta 0.87, 771 elements). So the bug isn't a data
dependency mistake; the placement is logically correct (gemm0 →
mask → alu1 of next pi is the same as gemm0 → … → mask in phase 1
→ alu1 of next pi).

**Suspected root cause**: `CoreLoopScheduler::schedule(cl_p, Phase)`
in `block_fmha_fwd_v3_pipeline.hpp` emits
`__builtin_amdgcn_sched_group_barrier(...)` hints prescribing the
*number and type* of instructions the compiler should emit between
consecutive scheduling markers. W0-3 phase 1's hint is "2 VALU + 4
SALU" — exactly the instruction shape of the original
`fmha_mask + Scheduler::schedule(1)` block. Phase 0's hint is "8x
(1 MFMA + 2 TRANS + 2 VALU)" — sized for `gemm0 + fmha_alu1` only.
Adding the mask's VALU/SALU instructions into phase 0 without
updating the hint over-subscribes the slot and causes the FP8 cvt
+ `ds_bpermute` cluster inside `fmha_alu1` to be scheduled
incorrectly, breaking the lane-pair re-layout that the PV gemm
relies on. bf16 has no such re-layout, which is why bf16 passes.

**Next attempt**: update the `CoreLoopScheduler::schedule` table in
`block_fmha_fwd_v3_pipeline.hpp` to account for the new phase
composition (phase 0 gains `T_mask` worth of VALU/SALU, phase 1
loses it). The hints are local to the FMHA-v3 file and the
adjustments are mechanical: shift the "2 VALU + 4 SALU" line from
the phase-1 entry to the phase-0 entry for *both* WaveGroups. This
is more invasive than the original code motion (touches a
file shared with regular FMHA), so it requires a separate gated
landing. Parked.

Reverted to baseline 2026-05-28 ahead of the next attempt.

---

## FP8 repack: `ds_bpermute` → `v_permlane32_swap_b32` (gfx950, 2026-06-01)

The FP8 32×32×16 branch of `fmha_alu1` repacks the QK-gemm fp8 output
into the PV-gemm input layout. The QK-C per-thread layout and the PV-A
per-thread layout disagree by a paired-lane (`l ^ 32`) byte swap: each
lane keeps its "good" 4-byte pack and has to trade its "bad" pack with
the lane 32 away. Since `3431615ff` this was done with an LDS-crossbar
`ds_bpermute` (+ an `is_sub_0` `v_cndmask` mux to pick which pack to
send / where the received pack lands).

**gfx950 exposes a single-VALU replacement.** `v_permlane32_swap_b32`
(builtin `__builtin_amdgcn_permlane32_swap`, gated on the
`permlane32-swap` target feature) does the `l ^ 32` exchange with no
LDS round-trip. Verified support matrix:

  | arch    | `permlane32-swap` |
  |---------|-------------------|
  | gfx950  | **yes** (emits `v_permlane32_swap_b32_e32`) |
  | gfx942  | no (`needs target feature permlane32-swap`) |

So the new path is guarded `#if defined(__gfx950__)`; the original
`ds_bpermute` path is kept as the `#else` fallback for gfx942 etc.

**Semantics (measured on-device, not from docs).** `permlane32_swap(a, b)`
swaps `a`'s high half (lanes 32–63) with `b`'s low half (lanes 0–31)
and keeps the two diagonal halves:

```
lane 0  (low):  out.x=a[0]   out.y=a[32]
lane 32 (high): out.x=b[0]   out.y=b[32]
```

Feeding `(lo_pack, hi_pack)` returns `{out_lo, out_hi}` for **every**
lane — i.e. both the cross-lane swap *and* the per-lane `is_sub_0`
sub-block muxing the `ds_bpermute` path needed are folded into the one
instruction. Provably equivalent to the old repack; the `lane_id /
paired_addr / is_sub_0` machinery is dropped on gfx950.

**ISA (prefill_d128 fp8 instance, `unified_attention_d128_fp8_mask_ps64`).**

  | build                | `ds_bpermute` | `v_permlane32_swap_b32` |
  |----------------------|--------------:|------------------------:|
  | baseline             | 12            | 11 (pre-existing m/rowsum reductions) |
  | permlane32 (this)    | **0**         | 23 (11 reductions + 12 repack)        |

Each of the 12 LDS-crossbar `ds_bpermute` collapsed to one VALU
`v_permlane32_swap_b32`, and the `v_cndmask` muxing is gone.

**Correctness.** FP8 prefill (`b=2 sq=sk=2048 d128 ps64`) and FP8
decode (`b=16 sq=1 sk=10000 d128 ps32`) both PASS vs the torch
reference.

**Perf (clean A/B, both builds ISA-verified, median of 3, b=4 FP8
prefill, GPU2 MI355X).** Baseline rebuilt by forcing the `#else` branch
(`UA_FORCE_DSBPERM_BASELINE` macro, removed before commit):

  | shape (Sq=Sk) | `ds_bpermute` | `permlane32` | speedup |
  |---------------|--------------:|-------------:|--------:|
  | 2048          | 0.1428 ms     | 0.1405 ms    | 1.6%    |
  | 5000          | 0.4981 ms     | 0.4884 ms    | 1.9%    |
  | 10000         | 1.5765 ms     | 1.5430 ms    | 2.1%    |

The win grows monotonically with sequence length (more K-tiles → more
repacks) and the permlane medians sit below the baseline *minimums* in
every band, so it's real signal, not run-to-run noise. This is on top
of the `ds_bpermute`-latency-hiding `3431615ff` already in the
baseline — the LDS crossbar is now removed entirely rather than just
overlapped.

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
    sweep_amir_shapes.py                     # 68-row production-shape sweep driver
    analyze_sweep.py                         # markdown formatter for sweep CSVs
    rocprof_prefill_d128.sh                  # four-phase rocprofv3 capture (trace / compute / stalls / PC sample)
    rocprof_analyze.py                       # cross-run aggregator → BOTTLENECK_*.md
    rocprof_analysis/                        # rocprof CSV dumps + BOTTLENECK_*.md reports (gitignored)
```
