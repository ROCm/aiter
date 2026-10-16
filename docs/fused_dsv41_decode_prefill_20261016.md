# DSV4.1 decode profiling and BF16 KV prefill — 2026-10-16

The selected implementation reduces H64 paged decode latency by 22.33–24.01%
and H128 by 15.45–17.02% relative to the previous H128 optimization at
`7d26ce07f4d85381fe178d1a53316d554cc1c4aa`. Peak measured decode throughput is
411.95 TFLOPS. It also adds `aiter.fused_dsv41_bf16`, consuming a contiguous
BF16 KV workspace, and compares optimized FP8 and BF16 prefill paths.

## Decode implementation and measured bottlenecks

- H64 retains eight lane-specific scale bytes across QK instead of sixteen.
  B32/B128 and legacy raw/rotated-Q scratch falls from 84/96 to 32/16 bytes
  per thread. Generic-page scratch falls from 88/92 to 20/20.
- H128 masks scores using the validity flags already stored with the KV tile
  in LDS. This removes redundant global index reads and validity/address
  calculations. BF16 tile loads now write the same flags. H128 scratch is
  zero for legacy, B32/B128 and generic pages.
- H64/H128 pure-cache attention uses the specialized pipeline for all token
  counts. FP8 prefetch occurs before QK; H64 commits after QK and H128 after
  PV. BF16 loads directly to LDS at the corresponding stage. Mixed prefix
  plus BF16 extend retains the previous dispatch.
- BF16 H64/H128 kernels have zero scratch and 250/244 VGPRs respectively.
  Every kernel still uses 67,584 bytes LDS per block; no occupancy increase
  is claimed.

| Decode case | VMEM reads before / after | VALU int32 before / after | MFMA before / after | Wait cycles / wave cycles before / after |
|---|---:|---:|---:|---:|
| 0 paged | 880320 / 577920 | 6731200 / 1751680 | 2867200 / 2867200 | 47.28% / 39.98% |
| 2 paged | 1509120 / 990720 | 11539200 / 3002880 | 4915200 / 4915200 | 50.06% / 44.59% |
| 4 paged | 1372160 / 634880 | 4400640 / 3392000 | 5898240 / 5898240 | 50.99% / 42.96% |

Counters are medians of the final five profiled dispatches, collected in
separate rocprofv3 passes. H64 VMEM read instructions fall 34.35% and int32
VALU instructions about 74%. H128 VMEM reads fall 53.73%. MFMA counts are
unchanged. H128 L2 misses are effectively unchanged (2,261,172 → 2,260,922),
while H64 misses fall only about 4%. These results identify redundant
index/address work and spill traffic as major costs; they do not establish
that HBM bandwidth was saturated. Wait ratios are wave-cycle counters, not
a direct decomposition of wall-clock kernel time.

## Final decode comparison

Before means the fixed H128 `scales` build at `7d26ce07f`, not the older KV
addressing repair. Standalone is the fixed 2026-10-15 H64-optimized legacy
library. All five paths consume the same logical fixtures; physical layouts
are prepared before timing. S is each request's long-cache length.

| T | H | K | S | Before paged us | Optimized paged us | Reduction | Standalone us | Optimized legacy us | TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 640 | 8192 | 207.29 | 157.53 | 24.01% | 206.62 | 161.68 | 298.21 |
| 560 | 64 | 640 | 32768 | 211.15 | 161.40 | 23.56% | 209.88 | 164.33 | 291.06 |
| 960 | 64 | 640 | 8192 | 252.86 | 195.89 | 22.53% | 257.48 | 200.55 | 411.11 |
| 960 | 64 | 640 | 32768 | 255.37 | 198.35 | 22.33% | 260.40 | 202.74 | 405.99 |
| 320 | 128 | 1152 | 8192 | 282.68 | 234.58 | 17.02% | 288.48 | 245.96 | 411.95 |
| 320 | 128 | 1152 | 32768 | 293.30 | 247.99 | 15.45% | 297.31 | 254.84 | 389.68 |

Both optimized paged and legacy paths beat the standalone baseline in all
six cases in this run. Decode throughput is 291.06–411.95 TFLOPS, with peak
about 61.5% of FlashMLA's published 670 TFLOPS B200 decode figure. This is a
comparison with the published claim, not a same-system B200 measurement.

## BF16 prefill contract and reference

`fused_dsv41_bf16(q, kv, indices, indptr, cos, sin, positions, **kwargs)`
accepts BF16 `[S,512]` or `[S,1,512]` KV, int32 row indices and CSR pointers.
The singleton-head view is zero-copy. KV must already include RoPE; invalid
indices are masked. The function directly reads BF16 and performs no gather,
dequantization, layout conversion or intermediate allocation when outputs
are supplied. Q is head-major, the RoPE tables are BF16, and output is the
existing grouped MXFP8 format. This is the BF16 workspace strategy used by
FlashMLA/vLLM, with AITER's existing tensor and numerical contract.

Reference versions:

- [vLLM gather and prefill call chain](https://github.com/vllm-project/vllm/blob/9f6e18fe642a187bf5e8aa9a5bf21e96217fedb4/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py#L424).
- [Pinned gather/dequant implementation](https://github.com/vllm-project/vllm/blob/9f6e18fe642a187bf5e8aa9a5bf21e96217fedb4/vllm/models/deepseek_v41/common/ops/cache_utils.py).
- [FlashMLA fused prefill API](https://github.com/vllm-project/FlashMLA/blob/0eee43b12f034b657133cf2afca6a72ebb6efccf/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp#L105).

The benchmark executes the pinned vLLM Triton gather body unchanged. On
MI355X, tuning launch parameters reduces the two-launch preparation from
about 21.30 to 7.68 us for 8K and 134.31 to 13.88 us for 32K. The compressed
range uses 2,048/8,192 workers respectively, with one wave; the 128-row window
uses 128 workers with one wave. These choices are benchmark configurations,
not a general dispatch policy installed in vLLM.

## Optimized prefill comparison

These are synthetic one-request prefill chunks: T=128/512/4096, H=64/128,
S=8192/32768, and K=640 (512 sparse compressed rows plus 128 window rows).
Queries share one cache range, unlike the multi-request decode fixtures.
FP8 uses randomly permuted physical B128 pages, and BF16 is decoded from
the exact same FP8 bytes. This isolates input-path performance; it does not
measure the accuracy benefit of an originally unquantized BF16 cache.

FP8 and BF16 columns include Q RoPE, sparse attention, inverse output RoPE
and MXFP8 quantization. The final column times the combined gather/dequant
and fused-attention graph directly; it is not the sum of independent
medians. Positive reduction means the combined BF16 route is faster.

| T | H | S | FP8 us | BF16 us | Gather us | Gather + BF16 us | Pipeline reduction vs FP8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 8192 | 72.19 | 44.45 | 7.69 | 58.04 | 19.60% |
| 512 | 64 | 8192 | 93.76 | 57.56 | 7.68 | 70.03 | 25.31% |
| 4096 | 64 | 8192 | 699.79 | 427.12 | 7.68 | 438.17 | 37.39% |
| 128 | 128 | 8192 | 70.89 | 59.78 | 7.67 | 73.18 | -3.24% |
| 512 | 128 | 8192 | 143.19 | 122.09 | 7.69 | 135.28 | 5.52% |
| 4096 | 128 | 8192 | 1141.68 | 918.36 | 7.68 | 934.61 | 18.14% |
| 128 | 64 | 32768 | 72.42 | 44.68 | 13.95 | 58.38 | 19.39% |
| 512 | 64 | 32768 | 93.92 | 57.99 | 13.91 | 71.48 | 23.89% |
| 4096 | 64 | 32768 | 702.10 | 437.59 | 13.98 | 455.22 | 35.16% |
| 128 | 128 | 32768 | 72.34 | 59.94 | 13.93 | 73.83 | -2.06% |
| 512 | 128 | 32768 | 144.43 | 122.91 | 13.95 | 136.31 | 5.62% |
| 4096 | 128 | 32768 | 1165.98 | 926.47 | 13.96 | 944.27 | 19.01% |

H64 BF16 attention is about 38–39% faster than optimized FP8 in these cases;
including preparation, the reduction is about 19–37%. For H128, T=128 is
about 2–3% slower after preparation, T=512 is about 5.5–5.6% faster, and
T=4096 is about 18–19% faster. Shared BF16 preparation amortizes repeated FP8
decode/page-address work as query reuse grows. BF16 storage is 1,024 rather
than 528 bytes per row (about 1.94x), and the conversion path holds both the
FP8 cache and BF16 workspace. Small chunks can lose the attention saving
to preparation overhead.

Optimization was measured against the same-run functional build:

| T | H | S | Before FP8 us | Optimized FP8 us | Before BF16 us | Optimized BF16 us |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 8192 | 94.11 | 72.19 | 44.54 | 44.45 |
| 512 | 64 | 8192 | 119.62 | 93.76 | 57.56 | 57.56 |
| 4096 | 64 | 8192 | 954.53 | 699.79 | 420.23 | 427.12 |
| 128 | 128 | 8192 | 70.93 | 70.89 | 59.69 | 59.78 |
| 512 | 128 | 8192 | 143.03 | 143.19 | 122.13 | 122.09 |
| 4096 | 128 | 8192 | 1620.14 | 1141.68 | 1114.23 | 918.36 |
| 128 | 64 | 32768 | 94.72 | 72.42 | 44.68 | 44.68 |
| 512 | 64 | 32768 | 119.88 | 93.92 | 57.97 | 57.99 |
| 4096 | 64 | 32768 | 1019.18 | 702.10 | 433.52 | 437.59 |
| 128 | 128 | 32768 | 72.32 | 72.34 | 59.96 | 59.94 |
| 512 | 128 | 32768 | 145.06 | 144.43 | 121.92 | 122.91 |
| 4096 | 128 | 32768 | 1680.57 | 1165.98 | 1117.90 | 926.47 |

The all-token specialized schedule improves T4096 H128 BF16 by about
17.0–17.6%. H64 BF16 has no measured scheduling gain: small shapes are
essentially unchanged, and T4096 varies by about +0.9–1.6% in latency.
No H64 BF16 attention speedup from this scheduling change is claimed.
The H64 FP8 compact-scale and all-token dispatch changes improve its
functional baseline, while optimized gather benefits the BF16 pipeline.
These are selected measured optimizations, not an exhaustive tuning claim.

## Validation and reproducibility

- 219 native tests pass: 183 existing legacy/paged tests and 36 BF16 tests.
  Coverage includes independent dense oracles, all head counts, raw/rotated
  Q, invalid interior tiles, empty caches/rows, boundary token counts,
  non-default streams, and poisoned output buffers before graph replay.
- The successful regression is retained because source SHA256s and JIT
  timestamps prove that the measured build is unchanged.
- All 20 pinned vLLM writer cases pass with zero error; legacy low-level
  default arguments pass.
- Six decode cases have bit-exact output/scale, max/LSE agreement, 18 dense
  query oracles, 1,500 event samples and 300 poison/replay checks.
- Twelve prefill cases have bit-exact FP8/BF16 output/scale, max/LSE and 36
  sampled dense oracles, 4,320 event samples and 864 poison/replay checks.
- Decode uses 8 calls/graph × 10 replays/sample; prefill uses 4 × 8. Both use
  five samples per visit and all forward/reverse cyclic implementation
  orders, reporting the median of visit medians. MI355X HIP6 was idle at
  preflight; clocks were not locked. Small percentage differences can vary.
- Black, Ruff, clang-format and `git diff --check` pass.

The timing excludes producer GEMMs, standalone Q rearrangement, metadata
construction and the rest of model forward. Q can be produced in the target
layout by its producer. Cache writing is excluded; gather/dequant is included
only in the pipeline column. This work does not integrate the full mega ABI,
dual independent cache arguments, Q16/O32, FP32 RoPE/epilogue, packed scale
token slices or full ATOM/vLLM model forward. Those remain separate work.

Remote workspace: `/home/jun_chen2_qle/yzhou_aiter/aiter-decode-prefill-20261016`.
Evidence: `/home/jun_chen2_qle/yzhou_aiter/decode-prefill-results-20261016`.
Scripts and raw JSON/XML/counters are archived with a final source/evidence
SHA256 manifest. The separate experiment scripts retain the fixed absolute
paths for replay on this host. GPU jobs must be serialized for timing.
