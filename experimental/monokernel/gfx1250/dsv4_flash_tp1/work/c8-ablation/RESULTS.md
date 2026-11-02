# C8 group ablation — 2026-10-21

All timings are microseconds per eight-token decode batch, second round only. Native is the installed ATOM/AITER dispatch. Jobs distributed across four GPUs.

| Native operation group | Isolated monokernel group | Native µs | Fused µs |
|---|---|---:|---:|
| mHC pre + RMSNorm | `mhc_batch` | 8.04 | 7.90 |
| Input quant + QA/KV + norms + QB + Q norm/RoPE | `qkv_batch` | 27.48 | 20.34 |
| SWA attention + split-KV reduce (128 entries) | `batch` | 7.61 | 39.78 |
| HCA attention + reduce (136 entries, envelope144) | `batch` | 8.00 | 38.98 |
| CSA attention + reduce (384 entries, envelope640) | `batch` | 9.70 | 36.46 |
| CSA attention + reduce (512 entries, envelope640) | `batch` | 10.83 | 35.01 |
| CSA attention + reduce (640 entries) | `batch` | 9.57 | 34.70 |
| Inverse RoPE + quant + OA + quant + OB | `batched` | 62.43 | 23.87 |
| Attention mHC post + FFN mHC pre/RMSNorm | `post_pre` | 5.01 | 11.65 |
| Router GEMM + shared expert quant/up/SwiGLU/down | `region` | 51.06 | 17.92 |
| Top6 selection + expert routing metadata | `run` | 7.87 | 12.05 |
| Input MXFP8 quant + routed MoE up/SwiGLU/down/reduce | `moe_region` | 48.71 | 76.79 |
| Shared+routed sum + mHC post | `combine` | 5.02 | 3.25 |
| Identity CSA selection + index translation (384) | `run` | 2.55 | 1.33 |

Compression projection + pool/norm/RoPE/cache/state writes:

| Variant | Native non-boundary / boundary µs | Fused non-boundary / boundary µs |
|---|---:|---:|
| Main ratio4 | 7.03 / 10.06 | 7.16 / 9.55 |
| Main ratio128 | 7.32 / 10.31 | 6.69 / 19.90 |
| Indexer ratio4 FP8 | 5.78 / 7.19 | 6.55 / 8.77 |

Extra monokernel stages:

| Stage | µs |
|---|---:|
| Compact SWA cache + metadata | 6.19 |
| Compact HCA cache + metadata | 6.26 |
| Compact CSA cache + metadata (384 valid) | 16.52 |
| Residual unshuffle | 1.99 |
| Residual shuffle | 1.36 |

## Interpretation

Largest isolated regressions: attention (~24–32 µs), routed MoE (~28 µs), post/pre boundary (~6.6 µs), routing (~4.2 µs). Cache compaction adds ~6–17 µs. Ratio128 compressor loses ~9.6 µs at rare boundary tokens. These are tuning targets, not an additive attribution of the full model slowdown.

Attention uses 64 splits for every row: 8 × 4 head groups × 64 = 2048 tasks looped over 256 resident CTAs. Empty splits still initialize partials and reduction reads 64 splits. This is a concrete batching inefficiency worth tuning; no parameter changes were made in this study.

## Scope and limitations

- Isolated groups compile independently; their register pressure differs from full monokernel. Do not sum as full-layer latency.
- SWA/CSA/HCA attention rows are alternative layer types; no summation across alternatives.
- Native means installed ATOM/AITER native dispatch, including its existing fusions.
- Standalone pre/post rows overlap with combined post-pre; use the combined row at the attention-to-FFN boundary.
- Compression rows at position1024 are non-boundary, position1535 boundary; boundary frequencies 1/4 and 1/128.
- Routing benchmark uses learned no-bias top6 sqrtsoftplus; not hash-based routing.
- MoE weights are checkpoint layer0 all256experts, synthetic distinct rows; expert distribution is not a serving-averaged trace.
- Cache staging and residual layout rows are implementation overhead, not matching native computation groups.
- Small numerical differences retained in validation fields, not treated as exact equivalence. No new end-to-end accuracy/performance run.

The scripts import saved dependencies from /work/batch-full, /work/modelwide, /work/ffn-c1, /work/full-attention-native, and /work/atom-actual-microbench. Preserve the previous full resume archive alongside this new ablation directory. Exact native kernel names and all samples are in individual JSON files; profiler traces provide names, not the primary timings.

All experiment servers on ports11500/11600/11700/11800 were stopped before this study. All ablation workers have finished.
